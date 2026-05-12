#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/zero_dce_packet.h"

#include <opencv2/opencv.hpp>

#include <cstring>

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &dcePacket = static_cast<ZeroDcePacket &>(packet);

    if (dcePacket.input_image.empty())
    {
        dcePacket.is_valid_image = false;
        dcePacket.input_tensor.assign(input_width_ * input_height_ * 3, 0);
        return;
    }

    dcePacket.is_valid_image = true;

    cv::Mat resized;
    cv::resize(
        dcePacket.input_image,
        resized,
        cv::Size(static_cast<int>(input_width_), static_cast<int>(input_height_)),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    cv::Mat rgbImage;
    cv::cvtColor(resized, rgbImage, cv::COLOR_BGR2RGB);

    if (!rgbImage.isContinuous())
    {
        rgbImage = rgbImage.clone();
    }

    dcePacket.input_width = static_cast<int>(input_width_);
    dcePacket.input_height = static_cast<int>(input_height_);
    dcePacket.input_tensor.resize(input_width_ * input_height_ * 3);
    std::memcpy(dcePacket.input_tensor.data(), rgbImage.data, dcePacket.input_tensor.size());
}
