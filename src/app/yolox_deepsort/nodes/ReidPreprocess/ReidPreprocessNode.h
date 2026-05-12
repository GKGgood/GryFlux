#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class ReidPreprocessNode : public GryFlux::NodeBase
{
public:
    ReidPreprocessNode(int targetWidth, int targetHeight)
        : targetWidth_(targetWidth),
          targetHeight_(targetHeight)
    {
    }

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int targetWidth_;
    int targetHeight_;
};

} // namespace PipelineNodes
