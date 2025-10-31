#include "res_sender.h"

#include <algorithm>
#include <cmath>

#include "package.h"
#include "utils/logger.h"

namespace GryFlux {
namespace RealESRGAN {

std::shared_ptr<DataObject> ResSender::process(const std::vector<std::shared_ptr<DataObject>> &inputs) {
    if (inputs.size() != 3) {
        LOG.error("[RealESRGAN::ResSender] Invalid input size: %zu", inputs.size());
        return nullptr;
    }

    auto original = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
    auto preprocessed = std::dynamic_pointer_cast<ImagePackage>(inputs[1]);
    auto sr_package = std::dynamic_pointer_cast<SuperResolutionPackage>(inputs[2]);

    if (!original || !preprocessed || !sr_package) {
        LOG.error("[RealESRGAN::ResSender] Package cast failed");
        return nullptr;
    }

    const cv::Mat &sr_tensor = sr_package->get_tensor();
    if (sr_tensor.empty()) {
        LOG.error("[RealESRGAN::ResSender] Empty SR tensor");
        return nullptr;
    }

    const float letterbox_scale = preprocessed->get_scale();
    const int x_pad = preprocessed->get_x_pad();
    const int y_pad = preprocessed->get_y_pad();
    const int model_w = preprocessed->get_width();
    const int model_h = preprocessed->get_height();

    const int orig_w = original->get_width();
    const int orig_h = original->get_height();

    const float sr_scale = static_cast<float>(sr_tensor.rows) / static_cast<float>(model_h);
    const int crop_left = std::clamp(static_cast<int>(std::round(x_pad * sr_scale)), 0, sr_tensor.cols);
    const int crop_top = std::clamp(static_cast<int>(std::round(y_pad * sr_scale)), 0, sr_tensor.rows);
    const int crop_right = std::clamp(static_cast<int>(std::round((model_w - x_pad) * sr_scale)), 0, sr_tensor.cols);
    const int crop_bottom = std::clamp(static_cast<int>(std::round((model_h - y_pad) * sr_scale)), 0, sr_tensor.rows);

    const int crop_width = std::max(crop_right - crop_left, 1);
    const int crop_height = std::max(crop_bottom - crop_top, 1);
    cv::Rect crop_roi(crop_left, crop_top, crop_width, crop_height);
    cv::Mat sr_cropped = sr_tensor(crop_roi).clone();

    const int target_w = std::max(static_cast<int>(std::round(orig_w * sr_scale)), 1);
    const int target_h = std::max(static_cast<int>(std::round(orig_h * sr_scale)), 1);
    cv::Mat sr_resized;
    cv::resize(sr_cropped, sr_resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(sr_resized, &min_val, &max_val);

    cv::Mat sr_scaled;
    if (max_val <= 2.0) {
        sr_resized.convertTo(sr_scaled, CV_32FC3, 255.0);
    } else {
        sr_scaled = sr_resized.clone();
    }

    cv::Mat sr_clamped_high;
    cv::min(sr_scaled, 255.0, sr_clamped_high);

    cv::Mat sr_clamped;
    cv::max(sr_clamped_high, 0.0, sr_clamped);

    cv::Mat sr_uint8;
    sr_clamped.convertTo(sr_uint8, CV_8UC3);

    cv::Mat sr_bgr;
    cv::cvtColor(sr_uint8, sr_bgr, cv::COLOR_RGB2BGR);

    LOG.info("[RealESRGAN::ResSender] id=%d | orig=%dx%d | model=%dx%d | scale=%.6f | letterbox_scale=%.6f",
             original->get_id(), orig_w, orig_h, model_w, model_h, sr_scale, letterbox_scale);

    return std::make_shared<ImagePackage>(sr_bgr, original->get_id());
}

} // namespace RealESRGAN
} // namespace GryFlux
