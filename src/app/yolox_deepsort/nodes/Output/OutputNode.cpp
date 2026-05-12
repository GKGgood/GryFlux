#include "nodes/Output/OutputNode.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes
{

void OutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    const auto &p = static_cast<const TrackDataPacket &>(packet);
    LOG.info(
        "[OutputNode] Frame %d completed with %zu detections and %zu features",
        p.frame_id,
        p.detections.size(),
        p.active_reid_feature_count);
}

} // namespace PipelineNodes
