#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/yolox_packet.h"

#include <opencv2/opencv.hpp>

#include <cstring>
#include <stdexcept>

namespace YoloxNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<YoloxPacket &>(packet);

    if (p.originalImage.empty())
    {
        throw std::runtime_error("Preprocess: empty input image");
    }

    cv::Mat rgb;
    cv::cvtColor(p.originalImage, rgb, cv::COLOR_BGR2RGB);

    const int imgWidth = p.originalImage.cols;
    const int imgHeight = p.originalImage.rows;

    if (imgWidth == static_cast<int>(modelWidth_) && imgHeight == static_cast<int>(modelHeight_))
    {
        if (!rgb.isContinuous())
        {
            rgb = rgb.clone();
        }
        p.inputTensor.resize(modelWidth_ * modelHeight_ * 3);
        std::memcpy(p.inputTensor.data(), rgb.data, p.inputTensor.size());
        p.modelWidth = modelWidth_;
        p.modelHeight = modelHeight_;
        p.scale = 1.0f;
        p.xPad = 0;
        p.yPad = 0;
        return;
    }

    const float scale = std::min(static_cast<float>(modelWidth_) / static_cast<float>(imgWidth),
                                 static_cast<float>(modelHeight_) / static_cast<float>(imgHeight));
    const int resizedWidth = static_cast<int>(static_cast<float>(imgWidth) * scale);
    const int resizedHeight = static_cast<int>(static_cast<float>(imgHeight) * scale);
    const int xPad = (static_cast<int>(modelWidth_) - resizedWidth) / 2;
    const int yPad = (static_cast<int>(modelHeight_) - resizedHeight) / 2;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resizedWidth, resizedHeight));

    cv::Mat letterbox(static_cast<int>(modelHeight_), static_cast<int>(modelWidth_), CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(xPad, yPad, resizedWidth, resizedHeight)));
    if (!letterbox.isContinuous())
    {
        letterbox = letterbox.clone();
    }

    p.inputTensor.resize(modelWidth_ * modelHeight_ * 3);
    std::memcpy(p.inputTensor.data(), letterbox.data, p.inputTensor.size());

    p.modelWidth = modelWidth_;
    p.modelHeight = modelHeight_;
    p.scale = scale;
    p.xPad = xPad;
    p.yPad = yPad;
}

} // namespace YoloxNodes
