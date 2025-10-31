#pragma once

#include "framework/data_consumer.h"
#include "utils/logger.h"
#include "utils/unified_allocator.h"

#include <atomic>
#include <filesystem>
#include <string>

namespace GryFlux
{
    class WriteConsumer : public DataConsumer
    {
    public:
        WriteConsumer(StreamingPipeline &pipeline,
                      std::atomic<bool> &running,
                      CPUAllocator *allocator,
                      std::string_view outputDir = "./outputs");

        int getProcessedFrames() const { return processedFrames_; }

    protected:
        void run() override;

    private:
        int processedFrames_;
        std::string outputDir_;
    };
}
