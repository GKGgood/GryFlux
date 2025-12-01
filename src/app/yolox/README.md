# GryFlux YOLOX Stream on Orin

本项目展示了如何在 NVIDIA Orin 平台上部署 YOLOX 目标检测模型。项目包含从 ONNX 模型转换到 TensorRT Engine（FP16/INT8）、C++ 流式推理框架的运行以及精度与速度的对比分析。

## 📋 目录

- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [模型转换与量化](#模型转换与量化)
  - [ONNX 转 FP16 Engine](#1-onnx-转-fp16-engine)
  - [ONNX 转 INT8 Engine (量化)](#2-onnx-转-int8-engine-量化)
- [编译与运行](#编译与运行)
- [配置选项](#配置选项)
- [精度验证与对比](#精度验证与对比)
- [性能基准](#性能基准)

## 🗂️ 项目结构

```
GryFlux/
├── scripts/                  # 脚本工具目录
│   ├── calib.py             # TensorRT INT8 量化校准脚本
│   ├── run_onnx_inference.py # ONNX 模型精度验证与基准生成
│   ├── compare_detections.py # 精度对比工具 (Engine vs ONNX)
│   └── detection_utils.py    # 检测相关通用工具函数
├── src/                      # C++ 源码目录
│   ├── app/
│   │   └── yolox/           # YOLOX 流式推理应用
│   │       ├── yolox_stream.cpp # 主程序入口
│   │       └── tasks/       # 任务节点实现 (预处理、推理、后处理)
│   └── framework/           # 流式框架核心 (Pipeline, TaskNode)
├── data/                     # 数据目录
│   ├── model/               # 模型文件 (ONNX, Engine)
│   └── dataset/             # 测试与校准数据集
├── logs/                     # 运行日志目录
├── outputs/                  # 对比报告输出目录
├── outputs_onnx/             # ONNX 基准结果目录
└── CMakeLists.txt           # 构建配置
```

## 🛠️ 环境准备

确保您的 Orin 环境已安装以下基础库：
- **TensorRT** (建议 8.5+)
- **CUDA / cuDNN**
- **OpenCV**
- **CMake** (3.10+)
- **Python 3** (用于运行转换和对比脚本)

安装 Python 依赖：
```bash
pip install onnxruntime-gpu opencv-python numpy pycuda
```

## 🔄 模型转换与量化

项目默认模型路径结构：
```
data/
├── model/
│   ├── yolox_s.onnx          # 原始 ONNX 模型
│   ├── yolox_s_fp16.engine   # 生成的 FP16 Engine
│   └── yolox_s_int8.engine   # 生成的 INT8 Engine
└── dataset/                  # 测试/校准数据集
```

### 1. ONNX 转 FP16 Engine

使用 TensorRT 自带的 `trtexec` 工具进行快速转换。FP16 精度通常能在几乎不损失精度的情况下带来显著的加速。

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=data/model/yolox_s.onnx \
    --saveEngine=data/model/yolox_s_fp16.engine \
    --fp16
```

### 2. ONNX 转 INT8 Engine (量化)

使用提供的 Python 脚本 `scripts/calib.py` 进行校准和量化。该脚本会读取 `data/dataset` 中的图片计算校准表。

```bash
python3 scripts/calib.py
```
*注意：请确保 `scripts/calib.py` 中的 `ONNX_PATH` 和 `CALIB_IMG_DIR` 配置正确。*

## 🚀 编译与运行

本项目使用 C++ 实现了一个高效的流式推理框架。

### 编译

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 运行推理

使用编译生成的 `yolox_stream` 可执行文件运行推理。程序将加载 Engine 模型并对数据集中的图片进行推理。

**运行 FP16 模型：**
```bash
./src/app/yolox/yolox_stream ../data/model/yolox_s_fp16.engine ../data/dataset
```

**运行 INT8 模型：**
```bash
./src/app/yolox/yolox_stream ../data/model/yolox_s_int8.engine ../data/dataset
```

运行日志将保存在 `logs/` 目录下，格式为 `StreamingExample-YYYYMMDD-HHMMSS.log`。

## 🔧 配置选项

### 量化配置
在 `scripts/calib.py` 中可以调整以下参数：
```python
ONNX_PATH = 'data/model/yolox_s.onnx'      # 原始模型路径
ENGINE_PATH = 'data/model/yolox_s_int8.engine' # 输出 Engine 路径
CALIB_IMG_DIR = 'data/dataset'             # 校准图片目录
CALIB_COUNT = 21                           # 校准使用的图片数量
INPUT_SHAPE = (1, 3, 640, 640)             # 模型输入尺寸
```

### 推理配置
在 `src/app/yolox/yolox_stream.cpp` 中可以调整 Pipeline 参数：
```cpp
// 注册 ObjectDetector 任务时的阈值参数
// ⚠️ 注意：此阈值应与 ONNX 验证脚本保持一致 (建议 0.25)
taskRegistry.registerTask<GryFlux::ObjectDetector>("objectDetector", 0.25f);

// Pipeline 线程数配置
GryFlux::StreamingPipeline pipeline(10); 
```

## 📊 精度验证与对比

为了验证 Engine 模型的精度，我们提供了一套完整的对比工具，将 Engine 的推理结果与 ONNX Runtime (CPU/CUDA) 的基准结果进行比对。

### 第一步：生成 ONNX 基准数据

运行 Python 脚本对数据集进行推理，生成基准 JSON 文件。

```bash
python3 scripts/run_onnx_inference.py \
    --model data/model/yolox_s.onnx \
    --dataset data/dataset \
    --output-dir outputs_onnx \
    --score-threshold 0.25
```
*输出文件位于 `outputs_onnx/detections.json`*

### 第二步：生成对比报告

使用 `compare_detections.py` 脚本解析 C++ 运行产生的日志文件，并与 ONNX 基准进行对比。

```bash
# 请替换 logs/ 下的实际文件名
python3 scripts/compare_detections.py \
    --reference outputs_onnx/detections.json \
    --fp16 logs/StreamingExample-FP16.log \
    --int8 logs/StreamingExample-INT8.log \
    --report-json outputs/comparison_report.json \
    --per-image-csv outputs/per_image_stats.csv
```

### 输出示例

脚本将输出详细的精度指标（Precision, Recall, F1-score, IoU）以及每一类的漏检情况。

```text
[TensorRT FP16] vs reference
  Precision        : 1.0000  (无误检)
  Mean IoU         : 0.9718  (定位极准)
  |BBox L1|        : 1.85 px (坐标误差极小)

[TensorRT INT8] vs reference
  Precision        : 0.9130
  Mean IoU         : 0.9160
  |BBox L1|        : 17.48 px
```

## 📈 性能基准

在 NVIDIA Orin 平台上，YOLOX-S 模型的典型性能表现如下：

| 模型格式 | 精度 (mAP) | 推理耗时 (ms) | 显存占用 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **ONNX (CPU)** | 基准 | ~150ms | - | 仅用于精度对齐 |
| **TRT FP16** | 保持原精度 | ~8ms | 低 | **推荐**，速度快且精度无损 |
| **TRT INT8** | 略有下降 | ~4ms | 极低 | 极致速度，需精细校准 |

### 常见问题

1. **召回率 (Recall) 低？**
   - 检查 C++ 代码 (`yolox_stream.cpp`) 中的 `threshold` 设置。如果 ONNX 脚本使用了 0.25 而 C++ 使用了 0.5，会导致大量低置信度目标被过滤，从而降低召回率。建议统一设置为 0.25。

2. **INT8 精度损失严重？**
   - 增加校准集的图片数量（建议 100-500 张）。
   - 确保校准时的预处理（Resize/Letterbox、归一化）与推理时完全一致。
