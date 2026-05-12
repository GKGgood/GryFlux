# GryFlux YOLOX

## 概述

`yolox` 用于执行基于 RKNN 的目标检测流水线。

- 输入：图片目录
- 输出：检测可视化图片
- DAG：`Input -> Preprocess -> Inference -> Postprocess -> Output`

## 目录说明
`
- `context/`：YOLOX 的 RKNN NPU 资源上下文
- `source/`：图片目录读取器
- `consumer/`：检测结果写出器
- `packet/`：流水线数据包定义
- `nodes/`：Input / Preprocess / Inference / Postprocess / Output 节点实现
- `3rdparty/`：本地部署依赖目录，仅用于本地构建和运行，不提交到 git

## 依赖准备

`yolox` 默认使用 `src/app/yolox/3rdparty` 下的依赖：

- OpenCV: `src/app/yolox/3rdparty/opencv`
- RKNN: `src/app/yolox/3rdparty/librknn_api`

## 构建方式

宿主构建：

```bash
cmake -S . -B build
cmake --build build --target yolox -j$(nproc)
```

交叉编译：

```bash
cmake -S . -B build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
  -DGRYFLUX_BUILD_PROFILING=1 \
  -DYOLOX_3RDPARTY_ROOT=/abs/path/to/yolox/3rdparty
cmake --build build-aarch64 --target yolox -j$(nproc)
```


## 运行方式

```bash
./yolox <model_path> <dataset_dir> [output_dir]
```

运行时只接收这三个位置参数。阈值、NPU 实例数、线程数等固定配置直接在 `src/app/yolox/yolox.cpp` 的 `CliOptions` 里修改后重新编译。

输入目录只扫描当前层级，支持 `.jpg/.jpeg/.png/.bmp`，按文件名排序。

## 输出说明

输出目录会包含一个子目录：

- `images/`：带检测框的可视化图片

当前实现默认按 YOLOX 常见三输出布局解码，并兼容单输出拼接形式的 RKNN 模型。
