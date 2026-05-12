#include "nodes/Inference/InferenceNode.h"

#include "context/fusion_npu_context.h"
#include "packet/fusionnetv2_packet.h"
#include "utils/logger.h"
#include <chrono>

namespace FusionNetV2Nodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<FusionNetV2Packet &>(packet);
    auto &npu = static_cast<FusionNpuContext &>(ctx);

    npu.setInputs(p.visYF32, p.infraredF32);
    npu.runInference();
    p.fusedYF32 = npu.getOutput(0);
}

} // namespace FusionNetV2Nodes
