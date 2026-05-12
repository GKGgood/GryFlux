#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/track_data_packet.h"

#include <opencv2/opencv.hpp>

#include <cstring>
#include <stdexcept>

namespace PipelineNodes
{

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<TrackDataPacket &>(packet);

    if (p.original_image.empty())
    {
        throw std::runtime_error("PreprocessNode: empty input image");
    }

    cv::Mat rgb;
    cv::cvtColor(p.original_image, rgb, cv::COLOR_BGR2RGB);

    const int imageWidth = p.original_image.cols;
    const int imageHeight = p.original_image.rows;

    if (imageWidth == static_cast<int>(modelWidth_) &&
        imageHeight == static_cast<int>(modelHeight_))
    {
        if (!rgb.isContinuous())
        {
            rgb = rgb.clone();
        }
        p.detection_input_tensor.resize(modelWidth_ * modelHeight_ * 3U);
        std::memcpy(
            p.detection_input_tensor.data(),
            rgb.data,
            p.detection_input_tensor.size());
        p.detection_model_width = modelWidth_;
        p.detection_model_height = modelHeight_;
        p.detection_scale = 1.0f;
        p.detection_x_pad = 0;
        p.detection_y_pad = 0;
        return;
    }

    const float scale = std::min(
        static_cast<float>(modelWidth_) / static_cast<float>(imageWidth),
        static_cast<float>(modelHeight_) / static_cast<float>(imageHeight));
    const int resizedWidth = static_cast<int>(static_cast<float>(imageWidth) * scale);
    const int resizedHeight = static_cast<int>(static_cast<float>(imageHeight) * scale);
    const int xPad = (static_cast<int>(modelWidth_) - resizedWidth) / 2;
    const int yPad = (static_cast<int>(modelHeight_) - resizedHeight) / 2;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resizedWidth, resizedHeight));

    cv::Mat letterbox(
        static_cast<int>(modelHeight_),
        static_cast<int>(modelWidth_),
        CV_8UC3,
        cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(xPad, yPad, resizedWidth, resizedHeight)));
    if (!letterbox.isContinuous())
    {
        letterbox = letterbox.clone();
    }

    p.detection_input_tensor.resize(modelWidth_ * modelHeight_ * 3U);
    std::memcpy(
        p.detection_input_tensor.data(),
        letterbox.data,
        p.detection_input_tensor.size());

    p.detection_model_width = modelWidth_;
    p.detection_model_height = modelHeight_;
    p.detection_scale = scale;
    p.detection_x_pad = xPad;
    p.detection_y_pad = yPad;
}

} // namespace PipelineNodes
