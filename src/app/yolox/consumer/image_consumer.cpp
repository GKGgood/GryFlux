#include "consumer/image_consumer.h"

#include "packet/yolox_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace
{
const char *kCocoClassNames[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

std::string classNameForId(int classId)
{
    constexpr int kClassCount = static_cast<int>(sizeof(kCocoClassNames) / sizeof(kCocoClassNames[0]));
    if (classId >= 0 && classId < kClassCount)
    {
        return kCocoClassNames[classId];
    }
    return "class_" + std::to_string(classId);
}

cv::Scalar colorForClass(int classId)
{
    static const cv::Scalar kColors[] = {
        cv::Scalar(255, 56, 56),   cv::Scalar(255, 157, 151), cv::Scalar(255, 112, 31),  cv::Scalar(255, 178, 29),
        cv::Scalar(207, 210, 49),  cv::Scalar(72, 249, 10),   cv::Scalar(146, 204, 23),  cv::Scalar(61, 219, 134),
        cv::Scalar(26, 147, 52),   cv::Scalar(0, 212, 187),   cv::Scalar(44, 153, 168),  cv::Scalar(0, 194, 255),
        cv::Scalar(52, 69, 147),   cv::Scalar(100, 115, 255), cv::Scalar(0, 24, 236),    cv::Scalar(132, 56, 255),
        cv::Scalar(82, 0, 133),    cv::Scalar(203, 56, 255),  cv::Scalar(255, 149, 200), cv::Scalar(255, 55, 199)};

    const int index = (classId >= 0) ? (classId % static_cast<int>(sizeof(kColors) / sizeof(kColors[0]))) : 0;
    return kColors[index];
}

std::string makeBaseName(const YoloxPacket &packet)
{
    if (!packet.imagePath.empty())
    {
        return fs::path(packet.imagePath).stem().string();
    }

    std::ostringstream oss;
    oss << "yolox_" << std::setfill('0') << std::setw(6) << packet.idx;
    return oss.str();
}

void drawDetections(cv::Mat &image, const std::vector<YoloxDetectionResult> &detections)
{
    for (const auto &det : detections)
    {
        const cv::Scalar color = colorForClass(det.classId);
        cv::rectangle(image, cv::Point(det.left, det.top), cv::Point(det.right, det.bottom), color, 2);

        std::ostringstream label;
        label << classNameForId(det.classId) << ' '
              << std::fixed << std::setprecision(2) << det.confidence;

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int textTop = std::max(det.top, textSize.height + 6);

        cv::rectangle(image,
                      cv::Point(det.left, textTop - textSize.height - 6),
                      cv::Point(det.left + textSize.width + 4, textTop),
                      color,
                      -1);
        cv::putText(image,
                    label.str(),
                    cv::Point(det.left + 2, textTop - 4),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255),
                    1);
    }
}
} // namespace

YoloxImageConsumer::YoloxImageConsumer(const std::string &outputDir)
    : outputDir_(fs::path(outputDir) / "images")
{
    if (outputDir.empty())
    {
        throw std::runtime_error("Output directory is empty");
    }

    fs::create_directories(outputDir_);
    LOG.info("Yolox image dir=%s", outputDir_.string().c_str());
}

void YoloxImageConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    if (!packet)
    {
        return;
    }

    auto &p = static_cast<YoloxPacket &>(*packet);

    if (p.originalImage.empty())
    {
        LOG.warning("Frame idx=%d has empty original image, skip image output", p.idx);
        return;
    }

    cv::Mat outputImage = p.originalImage.clone();
    drawDetections(outputImage, p.detections);

    const fs::path imagePath = outputDir_ / (makeBaseName(p) + ".jpg");
    if (!cv::imwrite(imagePath.string(), outputImage))
    {
        LOG.error("Failed to write result image: %s", imagePath.string().c_str());
        return;
    }

    writtenCount_.fetch_add(1, std::memory_order_relaxed);
}
