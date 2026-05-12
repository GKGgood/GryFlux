#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/zero_dce_packet.h"

#include <opencv2/opencv.hpp>

#include <vector>

namespace
{

cv::Mat convertOutputTensorToBgr(const ZeroDcePacket &packet)
{
    const std::size_t area =
        static_cast<std::size_t>(packet.output_height) * static_cast<std::size_t>(packet.output_width);
    const float *channelR = packet.output_tensor.data();
    const float *channelG = channelR + area;
    const float *channelB = channelG + area;

    std::vector<cv::Mat> rgbChannels;
    rgbChannels.reserve(3);
    rgbChannels.emplace_back(packet.output_height, packet.output_width, CV_32FC1, const_cast<float *>(channelR));
    rgbChannels.emplace_back(packet.output_height, packet.output_width, CV_32FC1, const_cast<float *>(channelG));
    rgbChannels.emplace_back(packet.output_height, packet.output_width, CV_32FC1, const_cast<float *>(channelB));

    cv::Mat outputRgbFloat;
    cv::merge(rgbChannels, outputRgbFloat);

    double maxVal = 0.0;
    cv::minMaxLoc(outputRgbFloat, nullptr, &maxVal);

    cv::Mat outputRgbU8;
    if (maxVal <= 1.5)
    {
        cv::max(outputRgbFloat, 0.0, outputRgbFloat);
        cv::min(outputRgbFloat, 1.0, outputRgbFloat);
        outputRgbFloat.convertTo(outputRgbU8, CV_8UC3, 255.0);
    }
    else
    {
        cv::max(outputRgbFloat, 0.0, outputRgbFloat);
        cv::min(outputRgbFloat, 255.0, outputRgbFloat);
        outputRgbFloat.convertTo(outputRgbU8, CV_8UC3);
    }

    cv::Mat outputBgr;
    cv::cvtColor(outputRgbU8, outputBgr, cv::COLOR_RGB2BGR);
    return outputBgr;
}

} // namespace

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &dcePacket = static_cast<ZeroDcePacket &>(packet);

    if (!dcePacket.is_valid_image)
    {
        return;
    }

    cv::Mat enhancedBgr = convertOutputTensorToBgr(dcePacket);

    if (dcePacket.input_image.cols != enhancedBgr.cols ||
        dcePacket.input_image.rows != enhancedBgr.rows)
    {
        cv::resize(
            enhancedBgr,
            dcePacket.output_image,
            dcePacket.input_image.size(),
            0.0,
            0.0,
            cv::INTER_LINEAR);
    }
    else
    {
        dcePacket.output_image = enhancedBgr;
    }
}
