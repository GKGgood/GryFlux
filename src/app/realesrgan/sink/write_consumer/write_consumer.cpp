#include "write_consumer.h"

#include <chrono>
#include <thread>

#include "package.h"

namespace GryFlux {
namespace RealESRGAN {

WriteConsumer::WriteConsumer(GryFlux::StreamingPipeline &pipeline,
                             std::atomic<bool> &running,
                             CPUAllocator *allocator,
                             std::string_view write_path)
    : GryFlux::DataConsumer(pipeline, running, allocator), processed_frames_(0) {
    if (!write_path.empty()) {
        std::filesystem::create_directories(write_path);
        output_path_ = write_path.data();
        LOG.info("[RealESRGAN::WriteConsumer] Output path set to: %s", write_path.data());
    } else {
        LOG.error("[RealESRGAN::WriteConsumer] Invalid output path");
    }
}

void WriteConsumer::run() {
    LOG.info("[RealESRGAN::WriteConsumer] Consumer started");

    while (shouldContinue()) {
        std::shared_ptr<DataObject> output;
        if (getData(output)) {
            auto result = std::dynamic_pointer_cast<ImagePackage>(output);
            if (!result) {
                continue;
            }

            ++processed_frames_;
            const cv::Mat &img = result->get_data();
            const std::string filename = output_path_ + "/sr_output_" + std::to_string(processed_frames_) + ".png";
            cv::imwrite(filename, img);
            LOG.info("[RealESRGAN::WriteConsumer] Frame %d processed", processed_frames_);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    LOG.info("[RealESRGAN::WriteConsumer] Processed frames: %d", processed_frames_);
    LOG.info("[RealESRGAN::WriteConsumer] Consumer finished");
}

} // namespace RealESRGAN
} // namespace GryFlux
