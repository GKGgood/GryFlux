#pragma once

#include "framework/node_base.h"

#include <cstddef>

namespace YoloxNodes
{

class PreprocessNode : public GryFlux::NodeBase
{
public:
    explicit PreprocessNode(std::size_t modelWidth, std::size_t modelHeight)
        : modelWidth_(modelWidth),
          modelHeight_(modelHeight)
    {
    }

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    std::size_t modelWidth_;
    std::size_t modelHeight_;
};

} // namespace YoloxNodes
