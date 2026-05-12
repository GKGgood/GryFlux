#include "nodes/Inference/InferenceNode.h"

#include "app/common/rknn_context.h"
#include "packet/realesrgan_packet.h"

#include <opencv2/opencv.hpp>

#include <stdexcept>
#include <vector>

namespace RealesrganNodes
{
namespace
{
cv::Mat makeOutputMatFromNCHW(const float *data, int channels, int height, int width)
{
    std::vector<cv::Mat> planes;
    planes.reserve(static_cast<std::size_t>(channels));

    const std::size_t planeSize = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    for (int i = 0; i < channels; ++i)
    {
        cv::Mat plane(height, width, CV_32F, const_cast<float *>(data + static_cast<std::size_t>(i) * planeSize));
        planes.push_back(plane.clone());
    }

    cv::Mat output;
    cv::merge(planes, output);
    return output;
}
} // namespace

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<RealesrganPacket &>(packet);
    auto &npu = static_cast<RKNNContext &>(ctx);

    p.srTensorF32.release();

    npu.setInput(p.inputTensor.data(), p.inputTensor.size());
    npu.runInference();

    if (npu.getOutputCount() != 1)
    {
        throw std::runtime_error("RealESRGAN expects single output tensor");
    }

    auto [outputPtr, outputElements] = npu.getOutput(0);
    const auto &attr = npu.getOutputAttr(0);

    const int channels = static_cast<int>(attr.dims[1]);
    const int height = static_cast<int>(attr.dims[2]);
    const int width = static_cast<int>(attr.dims[3]);
    if (channels <= 0 || height <= 0 || width <= 0)
    {
        throw std::runtime_error("Invalid output tensor shape");
    }

    std::vector<float> dequantized;
    const float *outputData = nullptr;
    if (attr.type == RKNN_TENSOR_INT8)
    {
        dequantized.resize(outputElements);
        auto *quantized = static_cast<int8_t *>(outputPtr);
        for (std::size_t i = 0; i < outputElements; ++i)
        {
            dequantized[i] = npu.deqntAffineToF32(quantized[i], attr.zp, attr.scale);
        }
        outputData = dequantized.data();
    }
    else
    {
        outputData = static_cast<float *>(outputPtr);
    }

    p.srTensorF32 = makeOutputMatFromNCHW(outputData, channels, height, width);
}

} // namespace RealesrganNodes
