#include "consumer/result_consumer.h"

#include "utils/logger.h"

#include <Eigen/Core>

#include <algorithm>
#include <stdexcept>

namespace
{

double normalizeFps(double fps)
{
    return fps > 0.0 ? fps : 25.0;
}

} // namespace

ResultConsumer::ResultConsumer(
    const std::string &outputPath,
    double fps,
    int width,
    int height)
    : tracker_(std::make_unique<DeepSortTracker>(0.4f, 100)),
      writer_(
          outputPath,
          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
          normalizeFps(fps),
          cv::Size(width, height))
{
    if (!writer_.isOpened())
    {
        throw std::runtime_error("Failed to create output video: " + outputPath);
    }

    LOG.info("[ResultConsumer] Writing YOLOX+DeepSORT output to %s", outputPath.c_str());
}

ResultConsumer::~ResultConsumer()
{
    if (writer_.isOpened())
    {
        writer_.release();
        LOG.info("[ResultConsumer] Output video finalized");
    }
}

void ResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    auto *trackPacket = static_cast<TrackDataPacket *>(packet.get());
    reorderBuffer_[trackPacket->frame_id] = std::move(packet);

    while (reorderBuffer_.count(expectedFrameId_) > 0)
    {
        auto currentPacket = std::move(reorderBuffer_[expectedFrameId_]);
        reorderBuffer_.erase(expectedFrameId_);
        processSequentialFrame(static_cast<TrackDataPacket *>(currentPacket.get()));
        ++expectedFrameId_;
    }
}

void ResultConsumer::processSequentialFrame(TrackDataPacket *packet)
{
    DETECTIONS trackerInput;
    const std::size_t usableCount =
        std::min(packet->detections.size(), packet->active_reid_feature_count);
    trackerInput.reserve(usableCount);

    if (packet->detections.size() != packet->active_reid_feature_count)
    {
        LOG.warning(
            "[ResultConsumer] Frame %d detection/feature count mismatch: %zu vs %zu",
            packet->frame_id,
            packet->detections.size(),
            packet->active_reid_feature_count);
    }

    for (std::size_t index = 0; index < usableCount; ++index)
    {
        const auto &detection = packet->detections[index];
        const float width = detection.x2 - detection.x1;
        const float height = detection.y2 - detection.y1;

        DETECTBOX box;
        box << detection.x1,
            detection.y1,
            width,
            height;

        const auto &featureData = packet->reid_features[index];
        FEATURE feature = Eigen::Map<const FEATURE>(featureData.data());
        trackerInput.emplace_back(box, detection.score, feature);
    }

    packet->active_tracks = tracker_->update(trackerInput);

    for (const auto &track : packet->active_tracks)
    {
        const auto tlwh = track.to_tlwh();
        const cv::Rect rect(
            static_cast<int>(tlwh(0)),
            static_cast<int>(tlwh(1)),
            static_cast<int>(tlwh(2)),
            static_cast<int>(tlwh(3)));
        cv::rectangle(packet->original_image, rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(
            packet->original_image,
            "ID: " + std::to_string(track.track_id),
            cv::Point(rect.x, std::max(0, rect.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2);
    }

    writer_.write(packet->original_image);
    ++writtenCount_;
    if (packet->frame_id % 30 == 0)
    {
        LOG.info("[ResultConsumer] Processed frame %d", packet->frame_id);
    }
}
