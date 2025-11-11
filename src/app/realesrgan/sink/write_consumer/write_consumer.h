#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "framework/data_consumer.h"
#include "utils/logger.h"
#include "utils/unified_allocator.h"

namespace GryFlux {
namespace RealESRGAN {

class WriteConsumer : public GryFlux::DataConsumer {
public:
    WriteConsumer(GryFlux::StreamingPipeline &pipeline,
                  std::atomic<bool> &running,
                  CPUAllocator *allocator,
                  std::string_view write_path = "./outputs");

    int getProcessedFrames() const { return processed_frames_; }

protected:
    void run() override;

private:
    int processed_frames_;
    std::string output_path_;
};

} // namespace RealESRGAN
} // namespace GryFlux
