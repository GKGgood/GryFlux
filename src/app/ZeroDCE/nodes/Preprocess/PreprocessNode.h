#pragma once

#include "framework/node_base.h"

#include <cstddef>

class PreprocessNode : public GryFlux::NodeBase
{
public:
    PreprocessNode(std::size_t inputWidth, std::size_t inputHeight)
        : input_width_(inputWidth),
          input_height_(inputHeight)
    {
    }

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    std::size_t input_width_;
    std::size_t input_height_;
};
