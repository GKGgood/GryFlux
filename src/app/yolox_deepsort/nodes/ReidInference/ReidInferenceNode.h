#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class ReidInferenceNode : public GryFlux::NodeBase
{
public:
    explicit ReidInferenceNode(int featureDimension)
        : featureDimension_(featureDimension)
    {
    }

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int featureDimension_;
};

} // namespace PipelineNodes
