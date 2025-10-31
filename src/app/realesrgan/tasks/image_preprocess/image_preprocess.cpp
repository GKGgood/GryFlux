#include "image_preprocess.h"

#include <cmath>

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

    const cv::Mat &frame_bgr = input->get_data();
    if (frame_bgr.empty()) {
        LOG.error("[RealESRGAN::ImagePreprocess] Empty input frame");
        return nullptr;
    }

    cv::Mat frame_rgb;
    cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

    const int orig_w = frame_rgb.cols;
    const int orig_h = frame_rgb.rows;

    if (orig_w == model_width_ && orig_h == model_height_) {
        LOG.info("[RealESRGAN::ImagePreprocess] Image already matches model input size");
        return std::make_shared<ImagePackage>(frame_rgb, input->get_id(), 1.0f, 0, 0);
    }

    const float scale = std::min(static_cast<float>(model_width_) / static_cast<float>(orig_w),
                                 static_cast<float>(model_height_) / static_cast<float>(orig_h));

    const int new_w = std::max(1, static_cast<int>(std::round(orig_w * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(orig_h * scale)));

    cv::Mat resized;
    cv::resize(frame_rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    cv::Mat letterbox(model_height_, model_width_, CV_8UC3, cv::Scalar(pad_value_, pad_value_, pad_value_));

    const int x_offset = (model_width_ - new_w) / 2;
    const int y_offset = (model_height_ - new_h) / 2;
    cv::Rect roi(x_offset, y_offset, new_w, new_h);
    resized.copyTo(letterbox(roi));

    LOG.info("[RealESRGAN::ImagePreprocess] Letterbox id=%d | orig=%dx%d -> scaled=%dx%d | scale=%.6f | pad=(%d,%d)",
             input->get_id(), orig_w, orig_h, new_w, new_h, scale, x_offset, y_offset);

    return std::make_shared<ImagePackage>(letterbox, input->get_id(), scale, x_offset, y_offset);
}

} // namespace RealESRGAN
} // namespace GryFlux
