#include "nodes/Inference/InferenceNode.h"

#include "app/common/rknn_context.h"
#include "packet/resnet_packet.h"

namespace ResnetNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<ResnetPacket &>(packet);
    auto &npu = static_cast<RKNNContext &>(ctx);

    npu.setInput(p.inputData.data(), p.inputData.size());
    npu.runInference();

    if (npu.getOutputCount() == 0)
    {
        throw std::runtime_error("Resnet inference produced no outputs");
    }

    auto [outputPtr, numElements] = npu.getOutput(0);
    const auto &attr = npu.getOutputAttr(0);
    if (p.logits.size() != numElements)
    {
        p.logits.resize(numElements);
    }

    if (attr.type == RKNN_TENSOR_INT8)
    {
        auto *quantData = static_cast<int8_t *>(outputPtr);
        for (std::size_t i = 0; i < numElements; ++i)
        {
            p.logits[i] = npu.deqntAffineToF32(quantData[i], attr.zp, attr.scale);
        }
    }
    else
    {
        auto *floatData = static_cast<float *>(outputPtr);
        for (std::size_t i = 0; i < numElements; ++i)
        {
            p.logits[i] = floatData[i];
        }
    }
}

} // namespace ResnetNodes
