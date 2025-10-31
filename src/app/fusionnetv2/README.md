# FusionNetV2 流式融合管线

该模块将 **FusionNetV2** 红外与可见光图像融合模型集成到 GryFlux 流式处理框架中，实现数据采集 → 预处理 → NPU 推理 → 融合重建 → 结果写出的全链路处理。

## 环境准备

- 已完成 GryFlux 工程的 CMake 配置（`cmake -S . -B build`）。
- 已安装 OpenCV、Rockchip RKNN 运行时，并在目标设备上可用。
- 目标 NPU 的 `fusionnetv2.rknn` 模型文件。
- 运行前请确保已加载 NPU 驱动：
  ```bash
  sudo modprobe rknpu        # 或 insmod 相应的 rknpu.ko
  ```
  若驱动未加载，程序会在初始化 `rknn_init` 时直接退出。

## 数据集组织方式

```
<dataset_root>/
  visible/     # 可见光彩色图像（BGR/RGB）
  infrared/    # 红外单通道灰度图像
```

要求：

1. 两个子目录名称固定为 `visible` 与 `infrared`（区分大小写）。
2. 两个目录内的图像文件需 **一一对应**，文件名（含扩展名）必须完全一致，例如 `00001.png` 同时存在于两侧。
3. 支持扩展名：`.png`、`.jpg`、`.jpeg`、`.bmp`。其他格式会被忽略。
4. 图像尺寸无限制，程序会自动缩放到模型输入尺寸（默认 640×480）。

## 构建步骤

```bash
rm -rf build && cmake -S . -B build
cmake --build build --target fusionnetv2_stream
```

> 若此前在其它路径生成过 `build/` 目录，请先清理或重新 `cmake -S . -B build`，以免缓存指向旧位置导致编译失败。例如：


## 运行示例

```bash
./build/src/app/fusionnetv2/fusionnetv2_stream \
  /path/to/fusionnetv2.rknn \
  /path/to/dataset/root \
  /path/to/output-dir
```

- `/path/to/fusionnetv2.rknn`：RKNN 模型文件，需与目标平台兼容。
- `/path/to/dataset/root`：符合上文组织方式的数据集根目录。
- `/path/to/output-dir`：融合结果输出目录，可缺省（默认 `./fusion_outputs`）。

程序运行时会在 `./logs` 下生成日志文件，并将融合后的彩色图像写入输出目录，命名格式为 `fusion_#.png`。

## 常见问题排查

| 现象 | 可能原因 | 排查建议 |
| ---- | -------- | -------- |
| `rknn_init failed with ret=-1` | 未加载 rknpu 驱动 | 使用 `lsmod | grep rknpu` 检查，必要时重新 `modprobe rknpu` |
| 输出图像偏暗 | 输入数据未归一化或模型输出为 0-1 范围 | 已在管线内部统一归一化；若仍有问题，请核对模型训练时的输入尺度 |
| 输出目录为空 | `visible/` 与 `infrared/` 文件名不匹配或图像损坏 | 检查日志，确认同名文件存在且可被 OpenCV 成功读取 |

如需自定义输入尺寸或输出格式，可在 `tasks/` 下的对应模块内修改参数并重新编译。
