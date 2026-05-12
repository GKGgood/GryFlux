#include "nodes/DetectionInference/DetectionInferenceNode.h"

#include "context/infercontext.h"
#include "packet/track_data_packet.h"

namespace PipelineNodes
{

void DetectionInferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<TrackDataPacket &>(packet);
    auto &npu = static_cast<InferContext &>(ctx);

    npu.setInput(p.detection_input_tensor.data(), p.detection_input_tensor.size());
    npu.runInference();

    const std::size_t outputCount = npu.getOutputCount();
    if (p.detection_outputs.size() != outputCount)
    {
        p.detection_outputs.resize(outputCount);
    }

    for (std::size_t i = 0; i < outputCount; ++i)
    {
        auto [outputPtr, numElements] = npu.getOutput(i);
        const auto &attr = npu.getOutputAttr(i);
        auto &output = p.detection_outputs[i];

        output.gridHeight = static_cast<std::size_t>(attr.dims[2]);
        output.gridWidth = static_cast<std::size_t>(attr.dims[3]);
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

} // namespace PipelineNodes
