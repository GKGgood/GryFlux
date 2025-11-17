# Real-ESRGAN 流式超分模块使用说明

## 1. 模块简介

本模块将 Real-ESRGAN RKNN 推理流程无缝集成进 GryFlux 流式计算框架，复用框架原有的生产者-任务-消费者结构，实现图像目录批量超分、自动信号同步和性能采样。管道结构与 `src/app/yolox` 保持一致，方便统一维护：

- `source/producer`：按文件名排序读取输入目录中的图像并注入管道。
- `tasks/image_preprocess`：检查尺寸、将图像规范化为 8-bit 三通道并执行 BGR→RGB 转换，匹配 `tools/inference_realesrgan.py` 里 RealESRGANer 的预处理。
- `tasks/rk_runner`：加载 RKNN 模型、执行推理、根据模型类型自动处理量化或浮点输入/输出。
- `tasks/res_sender`：将模型浮点输出裁剪至 [0,255] 范围、转成 `uint8` 并执行 RGB→BGR 回转，与参考脚本后处理保持一致。
- `sink/write_consumer`：使用输入同名文件将最终 BGR 结果落盘。

## 2. 运行环境准备

1. **编译工具链**
   - CMake ≥ 3.18
   - GCC/G++（建议 9.x 及以上）
2. **依赖库**
   - OpenCV（已在顶层 CMake 中通过 `find_package(OpenCV REQUIRED)` 自动链接）
   - Rockchip RKNN Runtime（确保目标设备已安装 `librknnrt.so`，默认搜索路径为 `/usr/lib/librknnrt.so`）
3. **模型文件**
   - 需准备 Real-ESRGAN RKNN 模型，例如 `realesrgan-x4-256.rknn`。模型输入为 256×256 RGB，输出为 4 倍超分的 1024×1024 图像。
4. **输入图像集合**
   - 将待处理图像放入同一目录。支持 `.jpg`、`.jpeg`、`.png`。

## 3. 编译步骤

假设工作目录为仓库根目录 `/home/hzhy/userdata/userdata/gxh/GryFlux`：

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --target realesrgan_stream -j$(nproc)
```

编译完成后可在 `build/src/app/realesrgan/` 下找到可执行文件 `realesrgan_stream`。

> 如需与其它模块同时编译，可直接执行 `cmake --build .`，所有应用会统一输出到对应子目录。

## 4. 运行方式

```bash
./src/app/realesrgan/realesrgan_stream <模型路径> <图像目录> [输出目录]
```

- `<模型路径>`：RKNN 模型绝对或相对路径，例如 `/data/models/realesrgan-x4-256.rknn`。
- `<图像目录>`：包含待处理图片的文件夹。例如 `./data/test_images`。
- `[输出目录]`（可选）：结果保存位置，默认 `./outputs`。目录不存在时会自动创建。

程序启动后会：

1. 初始化日志系统（日志默认写入 `./logs/`）。
2. 加载并固定绑定 RKNN 模型到 NPU（默认 core mask 使用 Core1，可在注册任务时自行调整）。
3. 按文件顺序处理输入目录中的图像，每张图像对应一次超分结果。
4. 将最终的 BGR 图片按输入同名文件写入输出目录（若原始文件名为空则退回 `sr_output_***.png`）。

## 5. PSNR 评估工具

项目提供了 `tools/psnr_eval.py`，用于比较参考高分辨率目录与推理输出目录之间的 PSNR。该脚本支持根据文件名后缀差异进行灵活匹配，示例命令如下：

```bash
python tools/psnr_eval.py \
   /root/workspace/resolution_high \
   /root/workspace/output_noquant \
   --match-delimiter _ \
   --match-drop-suffix-ref 1 \
   --match-drop-suffix-target 1
```

常用参数说明：

- `--match-delimiter`：按指定分隔符切分文件名主干（默认 `_`）。
- `--match-drop-suffix-ref/target`：分别丢弃参考目录与目标目录文件名末尾的若干段，以解决诸如 `_HR` vs `_LR` 的命名差异，未显式指定 target 时沿用 reference 的设置。
- `--match-ignore-case`：忽略大小写匹配。

以下对照表给出了近期测试所得的 PSNR（单位：dB），覆盖浮点 RKNN、量化 RKNN 以及 Python 参考实现三个输出目录：

| 图像 | output_pth | output_noquant | output_quant |
| --- | --- | --- | --- |
| img_006_SRF_4 | 19.0738 | 19.0757 | 18.9733 |
| img_014_SRF_4 | 18.8271 | 18.8293 | 18.7105 |
| img_016_SRF_4 | 24.5192 | 24.5232 | 24.1939 |
| img_023_SRF_4 | 20.7506 | 20.7547 | 20.3673 |
| img_068_SRF_4 | 24.4151 | 24.4191 | 24.0761 |
| img_081_SRF_4 | 26.3178 | 26.3188 | 25.8494 |
| img_082_SRF_4 | 24.1778 | 24.1831 | 24.0728 |
| img_084_SRF_4 | 22.6781 | 22.6835 | 22.4717 |
| img_087_SRF_4 | 22.9661 | 22.9687 | 22.6096 |
| img_100_SRF_4 | 20.8183 | 20.8192 | 20.6649 |
| **平均值** | **22.4544** | **22.4575** | **22.1989** |

> 注：Real-ESRGAN 以感知质量为优化目标，PSNR 通常低于以 L1/L2 为主的模型，表格数据仅用于不同部署路径的相对对比。

## 6. 常见问题排查

- **模型加载失败**：检查路径是否正确，文件是否为 RKNN 模型，目标设备是否部署对应的 Runtime 版本。
- **推理尺寸不一致**：当前实现假设模型输入固定为 256×256（不再自动缩放或补边）。若使用其它规格模型，请同步修改 `realesrgan_stream.cpp` 中注册任务的宽高参数以及预处理逻辑，或自行重新实现预处理步骤。
- **目录为空或格式不支持**：Producer 仅处理 `.jpg` / `.jpeg` / `.png`，其他格式会被忽略。
- **权限问题**：日志目录、输出目录均会在当前工作目录创建，确保具有写权限。

## 7. 二次开发指引

- 若需要切换到视频流或摄像头输入，可在 `source/producer` 中扩展新的 Producer 类型，保持输出 `ImagePackage` 即可。
- 如果希望在结果中叠加额外信息，可在 `tasks/res_sender` 中修改后处理逻辑，返回新的 `ImagePackage` 或自定义数据类型。
- 通过 `StreamingPipeline::enableProfiling(true)` 已启用性能统计，可在 `logs` 目录内查看各节点耗时。

祝使用顺利，如遇问题请结合日志信息定位或继续反馈。