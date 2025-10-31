#pragma once

#include "framework/data_object.h"
#include "opencv2/opencv.hpp"

namespace GryFlux
{
    class FusionImagePackage : public DataObject
    {
    public:
        FusionImagePackage(const cv::Mat &visible, const cv::Mat &infrared, int idx)
            : visible_(visible.clone()), infrared_(infrared.clone()), idx_(idx) {}

        const cv::Mat &get_visible() const
        {
            return visible_;
        }

        const cv::Mat &get_infrared() const
        {
            return infrared_;
        }

        int get_id() const
        {
            return idx_;
        }

        cv::Size get_visible_size() const
        {
            return visible_.size();
        }

    private:
        cv::Mat visible_;
        cv::Mat infrared_;
        int idx_;
    };

    class FusionPreprocessPackage : public DataObject
    {
    public:
        FusionPreprocessPackage(const cv::Mat &visY,
                                const cv::Mat &visCb,
                                const cv::Mat &visCr,
                                const cv::Mat &infrared,
                                cv::Size originalSize,
                                int idx)
            : visY_(visY.clone()), visCb_(visCb.clone()), visCr_(visCr.clone()), infrared_(infrared.clone()), originalSize_(originalSize), idx_(idx) {}

        const cv::Mat &get_vis_y() const { return visY_; }
        const cv::Mat &get_vis_cb() const { return visCb_; }
        const cv::Mat &get_vis_cr() const { return visCr_; }
        const cv::Mat &get_infrared() const { return infrared_; }
        cv::Size get_original_size() const { return originalSize_; }
        int get_id() const { return idx_; }

    private:
        cv::Mat visY_;
        cv::Mat visCb_;
        cv::Mat visCr_;
        cv::Mat infrared_;
        cv::Size originalSize_;
        int idx_;
    };

    class FusionRunnerPackage : public DataObject
    {
    public:
        FusionRunnerPackage(const cv::Mat &fusedY, int idx)
            : fusedY_(fusedY.clone()), idx_(idx) {}

        const cv::Mat &get_fused_y() const { return fusedY_; }
        int get_id() const { return idx_; }

    private:
        cv::Mat fusedY_;
        int idx_;
    };

    class FusionResultPackage : public DataObject
    {
    public:
        FusionResultPackage(const cv::Mat &image, int idx)
            : image_(image.clone()), idx_(idx) {}

        const cv::Mat &get_result() const { return image_; }
        int get_id() const { return idx_; }

    private:
        cv::Mat image_;
        int idx_;
    };
}
