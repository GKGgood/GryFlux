#include "image_preprocess.h"

#include "package.h"
#include "utils/logger.h"

namespace GryFlux {
namespace RealESRGAN {

std::shared_ptr<DataObject> ImagePreprocess::process(const std::vector<std::shared_ptr<DataObject>> &inputs) {
    if (inputs.size() != 1) {
        LOG.error("[RealESRGAN::ImagePreprocess] Invalid input size: %zu", inputs.size());
        return nullptr;
    }

    auto input = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
    if (!input) {
        LOG.error("[RealESRGAN::ImagePreprocess] Input cast failed");
        return nullptr;
    }

    const cv::Mat &frame = input->get_data();
    if (frame.empty()) {
        LOG.error("[RealESRGAN::ImagePreprocess] Empty input frame");
        return nullptr;
    }

    cv::Mat converted;
    switch (frame.channels()) {
        case 1:
            cv::cvtColor(frame, converted, cv::COLOR_GRAY2BGR);
            break;
        case 3:
            converted = frame;
            break;
        case 4:
            cv::cvtColor(frame, converted, cv::COLOR_BGRA2BGR);
            break;
        default:
            LOG.error("[RealESRGAN::ImagePreprocess] Unsupported channel count: %d", frame.channels());
            return nullptr;
    }

    if (converted.depth() != CV_8U) {
        cv::Mat converted_u8;
        converted.convertTo(converted_u8, CV_8U);
        converted = converted_u8;
    }

    if (converted.cols != model_width_ || converted.rows != model_height_) {
        LOG.error("[RealESRGAN::ImagePreprocess] Unexpected input size %dx%d, expected %dx%d",
                  converted.cols,
                  converted.rows,
                  model_width_,
                  model_height_);
        return nullptr;
    }

    cv::Mat frame_rgb;
    cv::cvtColor(converted, frame_rgb, cv::COLOR_BGR2RGB);

    return std::make_shared<ImagePackage>(frame_rgb, input->get_id(), input->get_filename());
}

} // namespace RealESRGAN
} // namespace GryFlux
