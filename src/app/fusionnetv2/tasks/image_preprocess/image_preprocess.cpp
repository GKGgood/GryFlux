#include "image_preprocess.h"

#include <opencv2/opencv.hpp>

#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    ImagePreprocess::ImagePreprocess(int modelWidth, int modelHeight)
        : modelWidth_(modelWidth), modelHeight_(modelHeight) {}

    std::shared_ptr<DataObject> ImagePreprocess::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        if (inputs.size() != 1)
        {
            LOG.error("[ImagePreprocess] Expected 1 input, got %zu", inputs.size());
            return nullptr;
        }

        auto imagePackage = std::dynamic_pointer_cast<FusionImagePackage>(inputs[0]);
        if (!imagePackage)
        {
            LOG.error("[ImagePreprocess] Invalid input package type");
            return nullptr;
        }

        const cv::Mat &visible = imagePackage->get_visible();
        const cv::Mat &infrared = imagePackage->get_infrared();
        if (visible.empty() || infrared.empty())
        {
            LOG.error("[ImagePreprocess] Empty input frames");
            return nullptr;
        }

        cv::Size targetSize(modelWidth_, modelHeight_);
        cv::Mat visibleResized;
        if (visible.size() != targetSize)
        {
            cv::resize(visible, visibleResized, targetSize, 0.0, 0.0, cv::INTER_LINEAR);
        }
        else
        {
            visibleResized = visible;
        }

        cv::Mat infraredResized;
        if (infrared.size() != targetSize)
        {
            cv::resize(infrared, infraredResized, targetSize, 0.0, 0.0, cv::INTER_LINEAR);
        }
        else
        {
            infraredResized = infrared;
        }

        cv::Mat ycrcb;
        cv::cvtColor(visibleResized, ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        if (channels.size() != 3)
        {
            LOG.error("[ImagePreprocess] Unexpected channel count: %zu", channels.size());
            return nullptr;
        }

    cv::Mat visYFloat;
    channels[0].convertTo(visYFloat, CV_32FC1, 1.0f / 255.0f);

    cv::Mat infraredFloat;
    infraredResized.convertTo(infraredFloat, CV_32FC1, 1.0f / 255.0f);

        return std::make_shared<FusionPreprocessPackage>(visYFloat,
                                 channels[2],
                                 channels[1],
                                 infraredFloat,
                                 visible.size(),
                                 imagePackage->get_id(),
                                 imagePackage->get_filename());
    }
}
