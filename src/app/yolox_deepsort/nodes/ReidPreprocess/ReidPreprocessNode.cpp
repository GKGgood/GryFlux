#include "nodes/ReidPreprocess/ReidPreprocessNode.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring>

namespace PipelineNodes
{

void ReidPreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<TrackDataPacket &>(packet);

    if (p.detections.empty())
    {
        p.active_reid_input_count = 0;
        p.active_reid_feature_count = 0;
        return;
    }

    const std::size_t cropCapacity = p.reid_input_tensors.size();
    p.active_reid_input_count = std::min(p.detections.size(), cropCapacity);
    if (p.detections.size() > cropCapacity)
    {
        LOG.warning(
            "[ReidPreprocessNode] Frame %d has %zu detections but only %zu ReID slots",
            p.frame_id,
            p.detections.size(),
            cropCapacity);
    }

    std::fill(p.reid_input_valid_flags.begin(), p.reid_input_valid_flags.end(), 0U);

    for (std::size_t index = 0; index < p.active_reid_input_count; ++index)
    {
        const auto &detection = p.detections[index];
        const int x1 = std::max(0, static_cast<int>(detection.x1));
        const int y1 = std::max(0, static_cast<int>(detection.y1));
        const int x2 = std::min(p.original_image.cols, static_cast<int>(detection.x2));
        const int y2 = std::min(p.original_image.rows, static_cast<int>(detection.y2));
        const int width = x2 - x1;
        const int height = y2 - y1;

        auto &buffer = p.reid_input_tensors[index];
        if (width <= 0 || height <= 0)
        {
            std::fill(buffer.begin(), buffer.end(), 0U);
            continue;
        }

        const cv::Mat crop = p.original_image(cv::Rect(x1, y1, width, height));

        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(targetWidth_, targetHeight_));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        if (!resized.isContinuous())
        {
            resized = resized.clone();
        }

        const std::size_t expectedBytes =
            static_cast<std::size_t>(targetWidth_) *
            static_cast<std::size_t>(targetHeight_) * 3U;
        if (buffer.size() != expectedBytes)
        {
            buffer.resize(expectedBytes);
        }
        std::memcpy(buffer.data(), resized.data, expectedBytes);
        p.reid_input_valid_flags[index] = 1U;
    }
}

} // namespace PipelineNodes
