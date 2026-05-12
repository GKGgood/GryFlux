#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/deeplab_packet.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <stdexcept>

namespace DeeplabNodes
{

namespace
{
constexpr int kClassCount = 21;

int clampInt(int value, int minValue, int maxValue)
{
    return std::max(minValue, std::min(value, maxValue));
}
} // namespace

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<DeeplabPacket &>(packet);

    if (p.inferenceOutputs.empty())
    {
        throw std::runtime_error("Postprocess: no inference outputs");
    }

    const auto &output = p.inferenceOutputs.front();
    if (output.data.empty() || output.gridH == 0 || output.gridW == 0)
    {
        throw std::runtime_error("Postprocess: invalid Deeplab output tensor");
    }


    cv::Mat mask(static_cast<int>(output.gridH), static_cast<int>(output.gridW), CV_8UC1);
    for (int y = 0; y < static_cast<int>(output.gridH); ++y)
    {
        uchar *maskPtr = mask.ptr<uchar>(y);
        for (int x = 0; x < static_cast<int>(output.gridW); ++x)
        {
            int bestClass = 0;
            float bestScore = output.data[(y * static_cast<int>(output.gridW) + x) * kClassCount];
            for (int c = 1; c < kClassCount; ++c)
            {
                const float score = output.data[(y * static_cast<int>(output.gridW) + x) * kClassCount + c];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }
            maskPtr[x] = static_cast<uchar>(bestClass);
        }
    }

    cv::Mat upsampledMask;
    cv::resize(mask,
               upsampledMask,
               cv::Size(static_cast<int>(p.modelWidth), static_cast<int>(p.modelHeight)),
               0,
               0,
               cv::INTER_NEAREST);

    const int cropX = clampInt(p.xPad, 0, upsampledMask.cols);
    const int cropY = clampInt(p.yPad, 0, upsampledMask.rows);
    const int cropW = clampInt(static_cast<int>(p.modelWidth) - cropX * 2, 1, upsampledMask.cols - cropX);
    const int cropH = clampInt(static_cast<int>(p.modelHeight) - cropY * 2, 1, upsampledMask.rows - cropY);

    const cv::Rect roi(cropX, cropY, cropW, cropH);
    const cv::Mat croppedMask = upsampledMask(roi);
    cv::resize(croppedMask, p.mask, p.originalImage.size(), 0, 0, cv::INTER_NEAREST);
}

} // namespace DeeplabNodes
