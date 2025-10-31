# Real-ESRGAN 流式超分模块使用说明

## 1. 模块简介

本模块将 Real-ESRGAN RKNN 推理流程无缝集成进 GryFlux 流式计算框架，复用框架原有的生产者-任务-消费者结构，实现图像目录批量超分、自动信号同步和性能采样。管道结构与 `src/app/yolox` 保持一致，方便统一维护：

- `source/producer`：按顺序读取输入目录中的图像并注入管道。
- `tasks/image_preprocess`：执行 RGB 转换与 letterbox 预处理（保持纵横比并记录 padding 信息）。
- `tasks/rk_runner`：加载 RKNN 模型、执行推理、完成量化反量化处理。
- `tasks/res_sender`：去除 padding、恢复超分图像尺寸并输出到下一节点。
- `sink/write_consumer`：将最终结果落盘。

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
4. 将最终的 BGR 图片写入输出目录，例如 `sr_output_001.png`。

## 5. 常见问题排查

- **模型加载失败**：检查路径是否正确，文件是否为 RKNN 模型，目标设备是否部署对应的 Runtime 版本。
- **推理尺寸不一致**：当前实现假设模型输入固定为 256×256。若使用其它规格模型，请同步修改 `realesrgan_stream.cpp` 中注册任务的宽高参数以及预处理逻辑。
- **目录为空或格式不支持**：Producer 仅处理 `.jpg` / `.jpeg` / `.png`，其他格式会被忽略。
- **权限问题**：日志目录、输出目录均会在当前工作目录创建，确保具有写权限。

## 6. 二次开发指引

- 若需要切换到视频流或摄像头输入，可在 `source/producer` 中扩展新的 Producer 类型，保持输出 `ImagePackage` 即可。
- 如果希望在结果中叠加额外信息，可在 `tasks/res_sender` 中修改后处理逻辑，返回新的 `ImagePackage` 或自定义数据类型。
- 通过 `StreamingPipeline::enableProfiling(true)` 已启用性能统计，可在 `logs` 目录内查看各节点耗时。

祝使用顺利，如遇问题请结合日志信息定位或继续反馈。