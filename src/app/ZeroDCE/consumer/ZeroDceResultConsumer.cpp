#include "consumer/ZeroDceResultConsumer.h"

#include "packet/zero_dce_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdexcept>

ZeroDceResultConsumer::ZeroDceResultConsumer(const std::string &outputDir,
                                             std::size_t totalFrames)
    : output_dir_(outputDir),
      total_frames_(totalFrames)
{
    if (outputDir.empty())
    {
        throw std::runtime_error("Output directory is empty");
    }

    fs::create_directories(output_dir_);
    LOG.info("ZeroDCE image dir=%s", output_dir_.string().c_str());
}

void ZeroDceResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    if (!packet)
    {
        return;
    }

    auto &dcePacket = static_cast<ZeroDcePacket &>(*packet);

    if (!dcePacket.is_valid_image)
    {
        LOG.warning("Frame idx=%llu has empty input image, skip image output",
                    static_cast<unsigned long long>(dcePacket.frame_id));
        const std::size_t completed = completed_frames_.fetch_add(1, std::memory_order_relaxed) + 1;
        std::cout << "\r[ZeroDCE][Consumer] progress: " << completed << " / " << total_frames_ << std::flush;
        return;
    }

    if (dcePacket.output_image.empty())
    {
        LOG.warning("Frame idx=%llu has empty output image, skip image output",
                    static_cast<unsigned long long>(dcePacket.frame_id));
        const std::size_t completed = completed_frames_.fetch_add(1, std::memory_order_relaxed) + 1;
        std::cout << "\r[ZeroDCE][Consumer] progress: " << completed << " / " << total_frames_ << std::flush;
        return;
    }

    const fs::path outPath = output_dir_ / fs::path(dcePacket.output_relative_path);
    fs::create_directories(outPath.parent_path());
    if (!cv::imwrite(outPath.string(), dcePacket.output_image))
    {
        LOG.error("Failed to write result image: %s", outPath.string().c_str());
        const std::size_t completed = completed_frames_.fetch_add(1, std::memory_order_relaxed) + 1;
        std::cout << "\r[ZeroDCE][Consumer] progress: " << completed << " / " << total_frames_ << std::flush;
        return;
    }

    written_frames_.fetch_add(1, std::memory_order_relaxed);

    const std::size_t completed = completed_frames_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::cout << "\r[ZeroDCE][Consumer] progress: " << completed << " / " << total_frames_ << std::flush;
}
