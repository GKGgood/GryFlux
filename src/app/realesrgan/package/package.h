#pragma once

#include <memory>

#include "framework/data_object.h"
#include "opencv2/opencv.hpp"

namespace GryFlux {
namespace RealESRGAN {

class ImagePackage : public GryFlux::DataObject {
public:
    ImagePackage(const cv::Mat &frame, int idx, float scale = 1.0f, int x_pad = 0, int y_pad = 0)
        : frame_(frame.clone()), idx_(idx), scale_(scale), x_pad_(x_pad), y_pad_(y_pad) {}

    const cv::Mat &get_data() const { return frame_; }
    cv::Mat &get_data() { return frame_; }
    int get_id() const { return idx_; }
    int get_width() const { return frame_.cols; }
    int get_height() const { return frame_.rows; }
    float get_scale() const { return scale_; }
    int get_x_pad() const { return x_pad_; }
    int get_y_pad() const { return y_pad_; }

private:
    cv::Mat frame_;
    int idx_;
    float scale_;
    int x_pad_;
    int y_pad_;
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
