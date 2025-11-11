#include "res_sender.h"

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

    static_cast<void>(preprocessed);

    const cv::Mat &sr_tensor = sr_package->get_tensor();
    if (sr_tensor.empty()) {
        LOG.error("[RealESRGAN::ResSender] Empty SR tensor");
        return nullptr;
    }

    cv::Mat sr_float;
    if (sr_tensor.type() != CV_32FC3) {
        sr_tensor.convertTo(sr_float, CV_32FC3);
    } else {
        sr_float = sr_tensor;
    }

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(sr_float, &min_val, &max_val);

    cv::Mat sr_scaled;
    if (max_val <= 2.0) {
        sr_float.convertTo(sr_scaled, CV_32FC3, 255.0);
    } else {
        sr_scaled = sr_float;
    }

    cv::Mat sr_clamped_high;
    cv::min(sr_scaled, 255.0, sr_clamped_high);

    cv::Mat sr_clamped;
    cv::max(sr_clamped_high, 0.0, sr_clamped);

    cv::Mat sr_uint8;
    sr_clamped.convertTo(sr_uint8, CV_8UC3);

    if (sr_uint8.channels() != 3) {
        LOG.error("[RealESRGAN::ResSender] Unexpected channel count in SR output: %d", sr_uint8.channels());
        return nullptr;
    }

    cv::Mat sr_bgr;
    cv::cvtColor(sr_uint8, sr_bgr, cv::COLOR_RGB2BGR);

    LOG.info("[RealESRGAN::ResSender] id=%d | input=%dx%d | output=%dx%d",
             original->get_id(),
             original->get_width(),
             original->get_height(),
             sr_bgr.cols,
             sr_bgr.rows);

    return std::make_shared<ImagePackage>(sr_bgr, original->get_id(), original->get_filename());
}

} // namespace RealESRGAN
} // namespace GryFlux
