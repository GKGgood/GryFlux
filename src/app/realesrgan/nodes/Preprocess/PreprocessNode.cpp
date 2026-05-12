#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/realesrgan_packet.h"

#include <opencv2/opencv.hpp>

#include <cstring>
#include <stdexcept>

namespace RealesrganNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealesrganPacket &>(packet);

    if (p.inputBgrU8.empty())
    {
        throw std::runtime_error("Preprocess: empty input image");
    }

    cv::Mat rgb;
    cv::cvtColor(p.inputBgrU8, rgb, cv::COLOR_BGR2RGB);

    if (p.inputBgrU8.cols == static_cast<int>(modelWidth_) &&
        p.inputBgrU8.rows == static_cast<int>(modelHeight_))
    {
        if (!rgb.isContinuous())
        {
            rgb = rgb.clone();
        }
        p.inputTensor.resize(modelWidth_ * modelHeight_ * 3);
        std::memcpy(p.inputTensor.data(), rgb.data, p.inputTensor.size());
        return;
    }

    cv::Mat resized;
    cv::resize(rgb,
               resized,
               cv::Size(static_cast<int>(modelWidth_), static_cast<int>(modelHeight_)));
    if (!resized.isContinuous())
    {
        resized = resized.clone();
    }

    p.inputTensor.resize(modelWidth_ * modelHeight_ * 3);
    std::memcpy(p.inputTensor.data(), resized.data, p.inputTensor.size());
}

} // namespace RealesrganNodes
