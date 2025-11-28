#include "write_consumer.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "package.h"

#include <opencv2/opencv.hpp>

namespace GryFlux
{
    WriteConsumer::WriteConsumer(StreamingPipeline &pipeline,
                                 std::atomic<bool> &running,
                                 CPUAllocator *allocator,
                                 std::string_view outputDir)
        : DataConsumer(pipeline, running, allocator),
          processedFrames_(0)
    {
        if (outputDir.empty())
        {
            LOG.error("[WriteConsumer] Output directory is empty");
            throw std::runtime_error("Invalid output directory");
        }

        outputDir_ = outputDir.data();
        std::filesystem::create_directories(outputDir_);
        LOG.info("[WriteConsumer] Output directory: %s", outputDir_.c_str());
    }

    void WriteConsumer::run()
    {
        LOG.info("[WriteConsumer] Consumer started");

        while (shouldContinue())
        {
            std::shared_ptr<DataObject> obj;
            if (getData(obj))
            {
                auto result = std::dynamic_pointer_cast<FusionResultPackage>(obj);
                if (!result)
                {
                    LOG.warning("[WriteConsumer] Received unexpected data type");
                    continue;
                }

                const auto &image = result->get_result();
                if (image.empty())
                {
                    LOG.warning("[WriteConsumer] Empty frame received");
                    continue;
                }

                ++processedFrames_;
                std::filesystem::path file = std::filesystem::path(outputDir_) /
                                              result->get_filename();
                cv::imwrite(file.string(), image);
                LOG.info("[WriteConsumer] Frame %d written to %s", processedFrames_, file.string().c_str());
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }

        LOG.info("[WriteConsumer] Processed %d frames", processedFrames_);
    }
}
