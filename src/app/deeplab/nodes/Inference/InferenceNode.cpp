#include "nodes/Inference/InferenceNode.h"

#include "app/common/rknn_context.h"
#include "packet/deeplab_packet.h"

#include <stdexcept>

namespace DeeplabNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<DeeplabPacket &>(packet);
    auto &npu = static_cast<RKNNContext &>(ctx);

    npu.setInput(p.inputData.data(), p.inputData.size());
    npu.runInference();

    const std::size_t outputCount = npu.getOutputCount();
    p.inferenceOutputs.resize(outputCount);

    for (std::size_t i = 0; i < outputCount; ++i)
    {
        auto [outputPtr, numElements] = npu.getOutput(i);
        const auto &attr = npu.getOutputAttr(i);
        auto &output = p.inferenceOutputs[i];

        output.gridH = static_cast<std::size_t>(attr.dims[1]);
        output.gridW = static_cast<std::size_t>(attr.dims[2]);
        output.channels = static_cast<std::size_t>(attr.dims[3]);
        if (output.data.size() != numElements)
        {
            output.data.resize(numElements);
        }

        if (attr.type == RKNN_TENSOR_INT8)
        {
            auto *quantData = static_cast<int8_t *>(outputPtr);
            for (std::size_t j = 0; j < numElements; ++j)
            {
                output.data[j] = npu.deqntAffineToF32(quantData[j], attr.zp, attr.scale);
            }
        }
        else
        {
            auto *floatData = static_cast<float *>(outputPtr);
            for (std::size_t j = 0; j < numElements; ++j)
            {
                output.data[j] = floatData[j];
            }
        }
    }
}

} // namespace DeeplabNodes
