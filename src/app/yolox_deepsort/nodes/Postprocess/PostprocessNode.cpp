#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace PipelineNodes
{

namespace
{

inline int clamp(float value, int minValue, int maxValue)
{
    return value > minValue ? (value < maxValue ? static_cast<int>(value) : maxValue)
                            : minValue;
}

} // namespace

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<TrackDataPacket &>(packet);

    auto &boxes = p.boxes;
    auto &scores = p.scores;
    auto &classIds = p.class_ids;

    boxes.clear();
    scores.clear();
    classIds.clear();
    p.detections.clear();

    int validCount = 0;
    for (std::size_t i = 0; i < p.detection_outputs.size(); ++i)
    {
        auto &output = p.detection_outputs[i];
        if (output.gridWidth == 0 || output.gridHeight == 0)
        {
            continue;
        }

        const int stride = static_cast<int>(p.detection_model_width / output.gridWidth);
        validCount += processScale(
            output.data.data(),
            static_cast<int>(output.gridHeight),
            static_cast<int>(output.gridWidth),
            stride,
            boxes,
            scores,
            classIds);
    }

    if (validCount <= 0)
    {
        return;
    }

    auto &indices = p.sorted_indices;
    if (static_cast<int>(indices.size()) < validCount)
    {
        indices.resize(static_cast<std::size_t>(validCount));
    }
    for (int i = 0; i < validCount; ++i)
    {
        indices[static_cast<std::size_t>(i)] = i;
    }

    quickSortIndicesInverse(scores, 0, validCount - 1, indices);

    std::set<int> classSet(classIds.begin(), classIds.end());
    for (int classId : classSet)
    {
        nms(validCount, boxes, classIds, indices, classId);
    }

    std::size_t detectionCount = 0;
    for (int i = 0; i < validCount && detectionCount < maxDetections_; ++i)
    {
        if (indices[static_cast<std::size_t>(i)] == -1)
        {
            continue;
        }

        const int idx = indices[static_cast<std::size_t>(i)];
        float x1 = boxes[static_cast<std::size_t>(idx) * 4U + 0U] -
                   static_cast<float>(p.detection_x_pad);
        float y1 = boxes[static_cast<std::size_t>(idx) * 4U + 1U] -
                   static_cast<float>(p.detection_y_pad);
        float x2 = x1 + boxes[static_cast<std::size_t>(idx) * 4U + 2U];
        float y2 = y1 + boxes[static_cast<std::size_t>(idx) * 4U + 3U];

        Detection detection;
        detection.x1 =
            static_cast<float>(clamp(x1, 0, static_cast<int>(p.detection_model_width))) /
            p.detection_scale;
        detection.y1 =
            static_cast<float>(clamp(y1, 0, static_cast<int>(p.detection_model_height))) /
            p.detection_scale;
        detection.x2 =
            static_cast<float>(clamp(x2, 0, static_cast<int>(p.detection_model_width))) /
            p.detection_scale;
        detection.y2 =
            static_cast<float>(clamp(y2, 0, static_cast<int>(p.detection_model_height))) /
            p.detection_scale;
        detection.score = scores[static_cast<std::size_t>(i)];
        detection.class_id = classIds[static_cast<std::size_t>(idx)];

        detection.x1 = std::max(0.0f, std::min(detection.x1, static_cast<float>(p.original_image.cols)));
        detection.y1 = std::max(0.0f, std::min(detection.y1, static_cast<float>(p.original_image.rows)));
        detection.x2 = std::max(0.0f, std::min(detection.x2, static_cast<float>(p.original_image.cols)));
        detection.y2 = std::max(0.0f, std::min(detection.y2, static_cast<float>(p.original_image.rows)));

        p.detections.push_back(detection);
        ++detectionCount;
    }

    LOG.info("[PostprocessNode] Frame %d detections=%zu", p.frame_id, p.detections.size());
}

int PostprocessNode::processScale(
    float *input,
    int gridH,
    int gridW,
    int stride,
    std::vector<float> &boxes,
    std::vector<float> &scores,
    std::vector<int> &classIds) const
{
    int validCount = 0;
    const int gridLen = gridH * gridW;

    for (int i = 0; i < gridH; ++i)
    {
        for (int j = 0; j < gridW; ++j)
        {
            const float boxConfidence = input[4 * gridLen + i * gridW + j];
            if (boxConfidence < confThreshold_)
            {
                continue;
            }

            const int offset = i * gridW + j;
            float *inPtr = input + offset;

            float boxX = inPtr[0];
            float boxY = inPtr[gridLen];
            float boxW = inPtr[2 * gridLen];
            float boxH = inPtr[3 * gridLen];

            boxX = (boxX + static_cast<float>(j)) * static_cast<float>(stride);
            boxY = (boxY + static_cast<float>(i)) * static_cast<float>(stride);
            boxW = std::exp(boxW) * static_cast<float>(stride);
            boxH = std::exp(boxH) * static_cast<float>(stride);
            boxX -= boxW / 2.0f;
            boxY -= boxH / 2.0f;

            float maxClassProb = inPtr[5 * gridLen];
            int maxClassId = 0;
            for (int k = 1; k < classCount_; ++k)
            {
                const float prob = inPtr[(5 + k) * gridLen];
                if (prob > maxClassProb)
                {
                    maxClassProb = prob;
                    maxClassId = k;
                }
            }

            if (maxClassProb > confThreshold_)
            {
                scores.push_back(maxClassProb * boxConfidence);
                classIds.push_back(maxClassId);
                boxes.push_back(boxX);
                boxes.push_back(boxY);
                boxes.push_back(boxW);
                boxes.push_back(boxH);
                ++validCount;
            }
        }
    }

    return validCount;
}

void PostprocessNode::quickSortIndicesInverse(
    std::vector<float> &input,
    int left,
    int right,
    std::vector<int> &indices) const
{
    if (left >= right)
    {
        return;
    }

    const float key = input[static_cast<std::size_t>(left)];
    const int keyIndex = indices[static_cast<std::size_t>(left)];
    int low = left;
    int high = right;

    while (low < high)
    {
        while (low < high && input[static_cast<std::size_t>(high)] <= key)
        {
            --high;
        }
        input[static_cast<std::size_t>(low)] = input[static_cast<std::size_t>(high)];
        indices[static_cast<std::size_t>(low)] = indices[static_cast<std::size_t>(high)];

        while (low < high && input[static_cast<std::size_t>(low)] >= key)
        {
            ++low;
        }
        input[static_cast<std::size_t>(high)] = input[static_cast<std::size_t>(low)];
        indices[static_cast<std::size_t>(high)] = indices[static_cast<std::size_t>(low)];
    }

    input[static_cast<std::size_t>(low)] = key;
    indices[static_cast<std::size_t>(low)] = keyIndex;

    quickSortIndicesInverse(input, left, low - 1, indices);
    quickSortIndicesInverse(input, low + 1, right, indices);
}

float PostprocessNode::calculateIoU(
    float xmin0,
    float ymin0,
    float xmax0,
    float ymax0,
    float xmin1,
    float ymin1,
    float xmax1,
    float ymax1) const
{
    const float w = std::max(0.0f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1) + 1.0f);
    const float h = std::max(0.0f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1) + 1.0f);
    const float intersect = w * h;
    const float unionArea =
        (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
        (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - intersect;
    return unionArea <= 0.0f ? 0.0f : (intersect / unionArea);
}

void PostprocessNode::nms(
    int validCount,
    const std::vector<float> &boxes,
    const std::vector<int> &classIds,
    std::vector<int> &order,
    int filterId) const
{
    for (int i = 0; i < validCount; ++i)
    {
        const int n = order[static_cast<std::size_t>(i)];
        if (n == -1 || classIds[static_cast<std::size_t>(n)] != filterId)
        {
            continue;
        }

        for (int j = i + 1; j < validCount; ++j)
        {
            const int m = order[static_cast<std::size_t>(j)];
            if (m == -1 || classIds[static_cast<std::size_t>(m)] != filterId)
            {
                continue;
            }

            const float xmin0 = boxes[static_cast<std::size_t>(n) * 4U + 0U];
            const float ymin0 = boxes[static_cast<std::size_t>(n) * 4U + 1U];
            const float xmax0 = boxes[static_cast<std::size_t>(n) * 4U + 0U] +
                                boxes[static_cast<std::size_t>(n) * 4U + 2U];
            const float ymax0 = boxes[static_cast<std::size_t>(n) * 4U + 1U] +
                                boxes[static_cast<std::size_t>(n) * 4U + 3U];

            const float xmin1 = boxes[static_cast<std::size_t>(m) * 4U + 0U];
            const float ymin1 = boxes[static_cast<std::size_t>(m) * 4U + 1U];
            const float xmax1 = boxes[static_cast<std::size_t>(m) * 4U + 0U] +
                                boxes[static_cast<std::size_t>(m) * 4U + 2U];
            const float ymax1 = boxes[static_cast<std::size_t>(m) * 4U + 1U] +
                                boxes[static_cast<std::size_t>(m) * 4U + 3U];

            const float iou = calculateIoU(
                xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > nmsThreshold_)
            {
                order[static_cast<std::size_t>(j)] = -1;
            }
        }
    }
}

} // namespace PipelineNodes
