#include "source/video_source.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <stdexcept>

VideoSource::VideoSource(
    const std::string &videoPath,
    std::size_t detectionCapacity,
    int reidWidth,
    int reidHeight,
    int reidFeatureDim)
    : capture_(videoPath),
      detectionCapacity_(detectionCapacity),
      reidWidth_(reidWidth),
      reidHeight_(reidHeight),
      reidFeatureDim_(reidFeatureDim)
{
    if (!capture_.isOpened())
    {
        throw std::runtime_error("Failed to open input video: " + videoPath);
    }

    setHasMore(true);
    readNextFrame();
    LOG.info(
        "[VideoSource] Opened %s, fps=%.2f, size=%dx%d",
        videoPath.c_str(),
        getFps(),
        getWidth(),
        getHeight());
}

VideoSource::~VideoSource()
{
    if (capture_.isOpened())
    {
        capture_.release();
    }
}

std::unique_ptr<GryFlux::DataPacket> VideoSource::produce()
{
    if (!hasMore())
    {
        return nullptr;
    }

    auto packet = std::make_unique<TrackDataPacket>(
        detectionCapacity_,
        reidWidth_,
        reidHeight_,
        reidFeatureDim_);
    packet->frame_id = frameId_++;
    packet->original_image = nextFrame_.clone();

    readNextFrame();
    return packet;
}

void VideoSource::readNextFrame()
{
    capture_ >> nextFrame_;
    if (!nextFrame_.empty())
    {
        return;
    }

    setHasMore(false);
    LOG.info("[VideoSource] Completed after %d frames", frameId_);
}
