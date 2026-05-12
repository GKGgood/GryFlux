#pragma once

#include "framework/data_consumer.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "No filesystem support found"
#endif

#include <atomic>
#include <string>

class ZeroDceResultConsumer : public GryFlux::DataConsumer
{
public:
    ZeroDceResultConsumer(const std::string &outputDir,
                          std::size_t totalFrames);

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    std::size_t getWrittenCount() const
    {
        return written_frames_.load(std::memory_order_relaxed);
    }

private:
    fs::path output_dir_;
    std::size_t total_frames_ = 0;
    std::atomic<std::size_t> completed_frames_{0};
    std::atomic<std::size_t> written_frames_{0};
};
