# GryFlux ZeroDCE

## 概述

`zero_dce` 用于执行基于 RKNN 的低照度图像增强流水线。

- 输入：图片目录
- 输出：增强结果图片
- DAG：`Preprocess -> Inference -> Postprocess`

## 目录说明

- `source/`：图片目录读取器
- `consumer/`：结果图片写出器
- `packet/`：流水线数据包定义
- `nodes/`：Preprocess / Inference / Postprocess 节点实现

## 依赖准备

`zero_dce` 默认复用 `src/app/realesrgan/3rdparty` 下的依赖：

- OpenCV: `src/app/realesrgan/3rdparty/opencv`
- RKNN: `src/app/realesrgan/3rdparty/librknn_api`

## 构建方式

宿主构建：

```bash
cmake -S . -B build
cmake --build build --target zero_dce -j$(nproc)
```

交叉编译：

```bash
cmake -S . -B build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/linaro-7.5-aarch64-linux-gnu.toolchain.cmake \
  -DGRYFLUX_BUILD_PROFILING=1 \
  -DZERODCE_3RDPARTY_ROOT=/abs/path/to/3rdparty
cmake --build build-aarch64 --target zero_dce -j$(nproc)
```

## 运行方式

```bash
./zero_dce_app <model_path> <input_dir> <output_dir>
```

运行时只接收这三个位置参数。NPU 实例数、线程数、最大并发包数等配置直接在 `src/app/ZeroDCE/zero_dce.cpp` 的 `AppConfig` 中修改后重新编译。

输入目录只扫描当前层级，支持 `.jpg/.jpeg/.png/.bmp`，按文件名排序。

## 输出说明

- 输出图按输入文件名写入输出目录
- consumer 保留终端进度条显示
