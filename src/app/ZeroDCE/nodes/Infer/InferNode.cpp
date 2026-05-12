#include "nodes/Infer/InferNode.h"

#include "app/common/rknn_context.h"
#include "packet/zero_dce_packet.h"

#include <stdexcept>

void InferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &dcePacket = static_cast<ZeroDcePacket &>(packet);
    if (!dcePacket.is_valid_image)
    {
        return;
    }

    auto &npu = static_cast<RKNNContext &>(ctx);

    npu.setInput(dcePacket.input_tensor.data(), dcePacket.input_tensor.size());
    npu.runInference();

    if (npu.getOutputCount() != 1)
    {
        throw std::runtime_error("ZeroDCE expects single output tensor");
    }

    auto [outputPtr, outputElements] = npu.getOutput(0);
    const auto &attr = npu.getOutputAttr(0);

    dcePacket.output_channels = static_cast<int>(attr.dims[1]);
    dcePacket.output_height = static_cast<int>(attr.dims[2]);
    dcePacket.output_width = static_cast<int>(attr.dims[3]);
    dcePacket.output_tensor.resize(outputElements);

    if (attr.type == RKNN_TENSOR_INT8)
    {
        auto *quantData = static_cast<int8_t *>(outputPtr);
        for (std::size_t i = 0; i < outputElements; ++i)
        {
            dcePacket.output_tensor[i] = npu.deqntAffineToF32(quantData[i], attr.zp, attr.scale);
        }
    }
    else
    {
        auto *floatData = static_cast<float *>(outputPtr);
        for (std::size_t i = 0; i < outputElements; ++i)
        {
            dcePacket.output_tensor[i] = floatData[i];
        }
    }
}
