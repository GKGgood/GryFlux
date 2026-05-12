#pragma once

#include "framework/data_source.h"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <string>

class VideoSource : public GryFlux::DataSource
{
public:
    VideoSource(
        const std::string &videoPath,
        std::size_t detectionCapacity,
        int reidWidth,
        int reidHeight,
        int reidFeatureDim);
    ~VideoSource() override;

    std::unique_ptr<GryFlux::DataPacket> produce() override;

    double getFps() const { return capture_.get(cv::CAP_PROP_FPS); }
    int getWidth() const { return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH)); }
    int getHeight() const { return static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT)); }

private:
    void readNextFrame();

    cv::VideoCapture capture_;
    cv::Mat nextFrame_;
    int frameId_ = 0;
    std::size_t detectionCapacity_ = 0;
    int reidWidth_ = 0;
    int reidHeight_ = 0;
    int reidFeatureDim_ = 0;
};
