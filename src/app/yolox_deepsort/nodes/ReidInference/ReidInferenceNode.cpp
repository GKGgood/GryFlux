#include "nodes/ReidInference/ReidInferenceNode.h"

#include "context/reid_context.h"
#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <algorithm>

namespace PipelineNodes
{

void ReidInferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<TrackDataPacket &>(packet);
    auto &npu = static_cast<ReidContext &>(ctx);

    p.active_reid_feature_count = p.active_reid_input_count;
    if (p.active_reid_input_count == 0)
    {
        return;
    }

    if (npu.getOutputCount() == 0)
    {
        LOG.error("[ReidInferenceNode] ReID model has no outputs");
        p.markFailed();
        return;
    }

    for (std::size_t index = 0; index < p.active_reid_input_count; ++index)
    {
        auto &feature = p.reid_features[index];
        if (feature.size() != static_cast<std::size_t>(featureDimension_))
        {
            feature.resize(static_cast<std::size_t>(featureDimension_), 0.0f);
        }

        if (!p.reid_input_valid_flags[index])
        {
            std::fill(feature.begin(), feature.end(), 0.0f);
            continue;
        }

        const auto &inputTensor = p.reid_input_tensors[index];
        npu.setInput(inputTensor.data(), inputTensor.size());
        npu.runInference();

        auto [outputPtr, numElements] = npu.getOutput(0);
        const auto &attr = npu.getOutputAttr(0);
        if (numElements < static_cast<std::size_t>(featureDimension_))
        {
            LOG.error(
                "[ReidInferenceNode] ReID output elements=%zu, required=%d",
                numElements,
                featureDimension_);
            p.markFailed();
            return;
        }

        if (attr.type == RKNN_TENSOR_INT8)
        {
            auto *quantData = static_cast<int8_t *>(outputPtr);
            for (int i = 0; i < featureDimension_; ++i)
            {
                feature[static_cast<std::size_t>(i)] =
                    npu.deqntAffineToF32(quantData[i], attr.zp, attr.scale);
            }
        }
        else
        {
            auto *floatData = static_cast<float *>(outputPtr);
            std::copy(
                floatData,
                floatData + featureDimension_,
                feature.begin());
        }
    }
}

} // namespace PipelineNodes
