#pragma once

#include <atomic>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "framework/data_producer.h"
#include "utils/logger.h"
#include "utils/unified_allocator.h"

namespace GryFlux
{
    class FusionImageProducer : public DataProducer
    {
    public:
        FusionImageProducer(StreamingPipeline &pipeline,
                            std::atomic<bool> &running,
                            CPUAllocator *allocator,
                            std::string_view datasetRoot,
                            std::size_t maxFrames = static_cast<std::size_t>(-1));
        ~FusionImageProducer();

    protected:
        void run() override;

    private:
        std::filesystem::path visiblePath_;
        std::filesystem::path infraredPath_;
        std::vector<std::string> fileList_;
        std::size_t maxFrames_;
        int frameCount_;

        static bool isImageFile(const std::filesystem::path &path);
    };
}
