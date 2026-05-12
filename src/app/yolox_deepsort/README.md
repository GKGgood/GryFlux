# GryFlux YOLOX DeepSORT

## 概述

`yolox_deepsort` 用于执行基于 RKNN 的 `YOLOX + ReID + DeepSORT` 视频多目标跟踪流水线。

- 输入：视频文件
- 输出：带轨迹 ID 的可视化视频
- DAG：`Input -> Preprocess -> DetectionInference -> Postprocess -> ReidPreprocess -> ReidInference -> Output`

## 目录说明

- `source/`：视频读取器
- `consumer/`：跟踪结果写出器
- `context/`：检测与 ReID 的 RKNN NPU 资源上下文
- `packet/`：流水线数据包定义
- `nodes/`：各阶段节点实现
- `utils/`：DeepSORT 相关算法实现
- `3rdparty/`：本地部署依赖目录，仅用于本地构建和运行，不提交到 git

## 依赖准备

`yolox_deepsort` 默认使用 `src/app/yolox_deepsort/3rdparty` 下的依赖：

- OpenCV: `src/app/yolox_deepsort/3rdparty/opencv`
- RKNN: `src/app/yolox_deepsort/3rdparty/librknn_api`
- Eigen: `src/app/yolox_deepsort/3rdparty/Eigen`

## 构建方式

```bash
cmake -S . -B build
cmake --build build --target yolox_deepsort -j$(nproc)
```

交叉编译：

```bash
cmake -S . -B build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
  -DGRYFLUX_BUILD_PROFILING=1 \
  -DYOLOX_DEEPSORT_3RDPARTY_ROOT=/abs/path/to/yolox_deepsort/3rdparty
cmake --build build-aarch64 --target yolox_deepsort -j$(nproc)
```

注意：

- 当前 target 链接的是 `aarch64` 的 `librknnrt.so`
- 需要在 RK3588 板端或对应交叉编译环境里完成最终链接和运行

## 运行方式

```bash
./yolox_deepsort <yolox_model> <reid_model> <input_video> [output_video]
```

运行时只接收位置参数。流水线参数固定写在 `src/app/yolox_deepsort/yolox_deepsort.cpp` 的 `CliOptions` 中，修改后需要重新编译。当前 DeepSORT 跟踪参数没有暴露到 `CliOptions`，而是写死在 `consumer/result_consumer.cpp` 的 `DeepSortTracker` 初始化里。

默认输出文件为 `./yolox_deepsort_output.mp4`。

## Node 作用说明

当前图执行顺序为：

`Input -> Preprocess -> DetectionInference -> Postprocess -> ReidPreprocess -> ReidInference -> Output`

各节点作用如下：

| Node | 作用 |
| --- | --- |
| `Input` | 从 `VideoSource` 取出一帧图像，写入 `TrackDataPacket`，作为整条流水线的原始输入。 |
| `Preprocess` | 对原始图像做检测模型输入预处理，包括 resize、padding、scale 记录和输入 tensor 填充，供 YOLOX 检测使用。 |
| `DetectionInference` | 调用检测 RKNN context 执行 YOLOX 推理，读取检测模型输出 tensor，并保存到 packet。 |
| `Postprocess` | 对 YOLOX 输出做解码、置信度过滤和 NMS，生成当前帧的检测框列表 `detections`。 |
| `ReidPreprocess` | 根据检测框从原图裁剪目标区域，缩放到 ReID 模型输入尺寸，生成每个目标的 ReID 输入 tensor。 |
| `ReidInference` | 逐个目标调用 ReID RKNN context，提取外观特征向量，写入 `reid_features`。 |
| `Output` | 作为图中的终点节点，表示当前 packet 已完成所有前序处理，便于统计和 profiling。 |

补充说明：

- `DeepSORT` 跟踪本身不在 `nodes/` 里完成，而是在 `consumer/result_consumer.cpp` 中执行。
- `ResultConsumer` 会按帧序取回完成的 packet，把检测框和 ReID 特征送入 `DeepSortTracker`，生成轨迹 ID，并把可视化结果写到输出视频。

## 默认参数

当前代码默认值如下。`CliOptions` 和 `ResultConsumer` 中记录的数值应与 README 保持一致。

### 基础流水线参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `classCount` | `80` | 检测类别数 |
| `reidFeatureDim` | `512` | ReID 特征维度 |
| `deviceId` | `0` | RKNN 运行设备 ID |
| `confThreshold` | `0.5` | 检测置信度阈值，用于过滤低分框 |
| `nmsThreshold` | `0.4` | NMS 阈值，用于抑制重复框 |
| `maxDetections` | `100` | 每帧最多保留的检测框数量 |
| `detectionNpuInstances` | `2` | YOLOX 检测 NPU 上下文实例数 |
| `reidNpuInstances` | `3` | ReID NPU 上下文实例数 |
| `threadPoolSize` | `8` | 框架线程池大小 |
| `maxActivePackets` | `4` | 同时处于流水线中的最大 packet 数 |
| `enableProfiling` | `true` | 是否启用 graph profiling |



## 输出说明

输出视频会在原始画面上叠加：

- 检测框
- 轨迹 ID

当前实现默认：

- 检测模型使用 YOLOX 常见三输出 RKNN 布局
- ReID 特征维度固定为 `512`
