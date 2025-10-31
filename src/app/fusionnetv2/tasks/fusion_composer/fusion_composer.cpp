#include "fusion_composer.h"

#include <vector>

#include <opencv2/opencv.hpp>

#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    std::shared_ptr<DataObject> FusionComposer::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        if (inputs.size() != 2)
        {
            LOG.error("[FusionComposer] Expected 2 inputs, got %zu", inputs.size());
            return nullptr;
        }

        auto preprocess = std::dynamic_pointer_cast<FusionPreprocessPackage>(inputs[0]);
        auto runnerOutput = std::dynamic_pointer_cast<FusionRunnerPackage>(inputs[1]);

        if (!preprocess || !runnerOutput)
        {
            LOG.error("[FusionComposer] Invalid input package types");
            return nullptr;
        }

        cv::Mat fusedY = runnerOutput->get_fused_y();
        if (fusedY.empty())
        {
            LOG.error("[FusionComposer] Empty fused Y channel");
            return nullptr;
        }

        cv::Mat fusedYScaled = fusedY.clone();
        double minVal = 0.0;
        double maxVal = 0.0;
        cv::minMaxLoc(fusedYScaled, &minVal, &maxVal);

        if (maxVal <= 1.5)
        {
            fusedYScaled *= 255.0f;
        }

        cv::max(fusedYScaled, 0.0f, fusedYScaled);
        cv::min(fusedYScaled, 255.0f, fusedYScaled);

        cv::Mat fusedYUint8;
        fusedYScaled.convertTo(fusedYUint8, CV_8UC1);

        std::vector<cv::Mat> ycrcb = {
            fusedYUint8,
            preprocess->get_vis_cr(),
            preprocess->get_vis_cb()};

        cv::Mat merged;
        cv::merge(ycrcb, merged);

        cv::Mat fusedBgr;
        cv::cvtColor(merged, fusedBgr, cv::COLOR_YCrCb2BGR);

        return std::make_shared<FusionResultPackage>(fusedBgr, runnerOutput->get_id());
    }
}
