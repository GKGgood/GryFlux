#pragma once

#include <memory>
#include <string>
#include <utility>

#include "framework/data_object.h"
#include "opencv2/opencv.hpp"

namespace GryFlux {
namespace RealESRGAN {

class ImagePackage : public GryFlux::DataObject {
public:
    ImagePackage(const cv::Mat &frame, int idx, std::string filename = {})
        : frame_(frame.clone()), idx_(idx), filename_(std::move(filename)) {}

    const cv::Mat &get_data() const { return frame_; }
    cv::Mat &get_data() { return frame_; }
    int get_id() const { return idx_; }
    int get_width() const { return frame_.cols; }
    int get_height() const { return frame_.rows; }
    const std::string &get_filename() const { return filename_; }

private:
    cv::Mat frame_;
    int idx_;
    std::string filename_;
};

class SuperResolutionPackage : public GryFlux::DataObject {
public:
    explicit SuperResolutionPackage(const cv::Mat &sr_tensor)
        : sr_tensor_(sr_tensor.clone()) {}

    const cv::Mat &get_tensor() const { return sr_tensor_; }
    cv::Mat &get_tensor() { return sr_tensor_; }
    int get_width() const { return sr_tensor_.cols; }
    int get_height() const { return sr_tensor_.rows; }

private:
    cv::Mat sr_tensor_;
};

} // namespace RealESRGAN
} // namespace GryFlux
