#include "image_producer.h"

#include <stdexcept>

namespace GryFlux {
namespace RealESRGAN {

ImageProducer::ImageProducer(GryFlux::StreamingPipeline &pipeline,
                             std::atomic<bool> &running,
                             CPUAllocator *allocator,
                             std::string_view dataset_path,
                             std::size_t max_frames)
    : GryFlux::DataProducer(pipeline, running, allocator),
      frame_count_(0),
      max_frames_(max_frames) {
    if (!std::filesystem::exists(dataset_path) || !std::filesystem::is_directory(dataset_path)) {
        LOG.error("Failed to open %s", dataset_path.data());
        throw std::runtime_error("wrong dataset path");
    }

    dataset_path_ = dataset_path;
}

ImageProducer::~ImageProducer() = default;

void ImageProducer::run() {
    LOG.info("[RealESRGAN::ImageProducer] Producer start");
    const bool unlimited = (max_frames_ == static_cast<std::size_t>(-1));
    for (auto &entry : std::filesystem::directory_iterator(dataset_path_)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string file_path = entry.path().string();
        if (file_path.find(".jpg") == std::string::npos && file_path.find(".png") == std::string::npos &&
            file_path.find(".jpeg") == std::string::npos) {
            continue;
        }

        cv::Mat src_frame = cv::imread(file_path);
        if (src_frame.empty()) {
            LOG.error("Failed to read image %s", file_path.c_str());
            continue;
        }

        auto input_data = std::make_shared<ImagePackage>(src_frame, frame_count_);
        if (!addData(input_data)) {
            LOG.error("[RealESRGAN::ImageProducer] Failed to add input data to pipeline");
            break;
        }

        ++frame_count_;
        if (!unlimited && static_cast<std::size_t>(frame_count_) >= max_frames_) {
            break;
        }
    }

    LOG.info("[RealESRGAN::ImageProducer] Producer finished, generated %d frames", frame_count_);
    stop();
}

} // namespace RealESRGAN
} // namespace GryFlux
