# FusionNetV2 图像融合流式模块说明

## 1. 模块概述

FusionNetV2 模块将可见光与红外图像的融合模型（RKNN 版本）集成进 GryFlux 流式计算框架，通过生产者-任务-消费者流水线完成批量图像读取、推理与写回。流水线节点如下：

- `FusionImageProducer`：从数据集目录中成对加载可见光与红外图像；
- `ImagePreprocess`：将图像缩放至 640×480，提取可见光 Y 通道并与红外图标准化为 `float32`；
- `RkRunner`：调用 RKNN 模型，支持浮点与量化权重；
- `FusionComposer`：将模型输出的融合 Y 通道与原始可见光的 Cb/Cr 通道重建为 BGR 图像；
- `WriteConsumer`：将融合后的结果写入输出目录。

## 2. 数据与目录约定

`FusionImageProducer` 假定数据集根目录下包含两个子文件夹：

```
<dataset_root>/visible/*.png  # 或 jpg/jpeg/bmp
<dataset_root>/infrared/*.png
```

两个子目录需使用一致的文件名，模块将只处理同时在可见光与红外目录中存在的文件。

输出目录中的结果与输入可见光图像同名（例如 `00004N.png`），方便和原图对照。

## 3. 环境准备

- CMake ≥ 3.18；
- GCC/G++（建议 9.x 及以上）；
- OpenCV（顶层 CMake 已通过 `find_package(OpenCV REQUIRED)` 引入）；
- Rockchip RKNN Runtime 1.6.x（需确保 `librknnrt.so` 已安装且可被运行时找到）。

## 4. 编译步骤

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --target fusionnetv2_stream -j$(nproc)
```

成功后，可执行文件位于 `build/src/app/fusionnetv2/fusionnetv2_stream`。

## 5. 运行方式

```bash
./src/app/fusionnetv2/fusionnetv2_stream <模型路径> <数据集根目录> [输出目录]
```

- `<模型路径>`：RKNN 模型文件，例如 `/data/models/fusionnetv2.rknn`；
- `<数据集根目录>`：包含 `visible/` 与 `infrared/` 子目录的路径；
- `[输出目录]`（可选）：保存融合结果的目录，缺省为 `./fusion_outputs`。

程序启动后会自动创建 `./logs`，日志文件命名为 `FusionNetV2Stream-*.log`，同时在终端输出进度。流水线默认启用性能采样，可在日志尾部的 “Pipeline Statistics” 查看各任务平均耗时。

## 6. 模型与预处理说明

- 预处理将可见光图像转换到 YCrCb 颜色空间，只保留 Y 通道供模型输入，同时将 Cb/Cr 通道原样保留；
- 模型输入输出均在 `float32` 范围 0~1（若模型属性指示量化，则会依 scale/zp 自动量化/反量化），适用于单输入或双输入模型（Y 通道和红外通道）；
- `FusionComposer` 会将模型输出的融合 Y 通道依据实际最大值自动判定是否需要乘以 255，再与原 Cb/Cr 组合并转换回 BGR；
- `WriteConsumer` 会持续写入 PNG 文件，并在日志中记录写出的帧序号与路径。

## 7. 指标评估

可使用仓库提供的 `tools/fusion_metrics.py` 对融合结果进行互信息（MI）、信息熵、平均差、空间频率等指标统计。示例：

```bash
python3 tools/fusion_metrics.py fusion_img/visible fusion_img/infrared fusion_img/fusion \
	--output-csv fusion_img/metrics.csv
```

下表展示了同一批次数据在量化/非量化模型上的平均指标（单位：互信息为 bit，平均差/空间频率沿用脚本输出单位）。

| 模型 | MI_visible | MI_infrared | MI_total | Entropy | StdDifference | SpatialFrequency |
| --- | --- | --- | --- | --- | --- | --- |
| PT (`genv2_100.pt`) | 3.954 | 0.517 | 4.471 | 6.5845 | 20.6526 | 11.2649 |
| 量化 (`fusionnetv2.rknn`) | 2.855 | 0.509 | 3.364 | 6.500 | 20.305050 | 11.489 |
| 非量化 (`fusionnetv2_noquant.rknn`) | 4.032 | 0.522 | 4.554 | 6.580 | 21.244 | 11.233 |

如需其他数据集或更多配置，可借助脚本的递归与文件名匹配参数（`--recursive`、`--match-drop-suffix-*` 等）灵活适配目录结构。

## 8. 故障排查

- **模型加载失败**：检查路径与权限，确认 RKNN Runtime 版本与模型兼容；
- **提示缺少图像对**：确认可见光与红外目录下文件名一致；
- **尺寸不匹配**：模块会自动 resize 到 640×480，如需其它分辨率可在 `fusionnetv2_stream.cpp` 中修改常量；
- **输出过暗或过亮**：确认模型输出范围是否为 0~1；若使用自训练模型，可根据需要调整 `FusionComposer` 的缩放逻辑。

## 9. 扩展建议

- 可在 `FusionImageProducer` 中添加更多数据过滤或排序逻辑；
- 若需额外的评价指标，可在 `WriteConsumer` 之外新增统计任务，沿用 GryFlux 的任务注册机制；
- 对于多模型部署，可在命令行外层编写脚本循环不同模型路径并汇总日志中的 `Task [rkRunner]` 耗时，以对比推理性能。

祝使用顺利，如有疑问可结合日志信息继续定位问题。
