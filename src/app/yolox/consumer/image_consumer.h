#pragma once

#include "framework/data_consumer.h"
#include "packet/yolox_packet.h"

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

class YoloxImageConsumer : public GryFlux::DataConsumer
{
public:
    explicit YoloxImageConsumer(const std::string &outputDir);

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    std::size_t getWrittenCount() const { return writtenCount_.load(std::memory_order_relaxed); }

private:
    fs::path outputDir_;
    std::atomic<std::size_t> writtenCount_{0};
};
