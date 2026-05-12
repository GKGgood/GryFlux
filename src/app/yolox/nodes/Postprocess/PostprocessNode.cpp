#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/yolox_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace YoloxNodes
{

namespace
{
constexpr int OBJ_CLASS_NUM = 80;

inline int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}
} // namespace

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<YoloxPacket &>(packet);

    auto &boxes = p.boxes;
    auto &scores = p.scores;
    auto &classIds = p.classIds;

    boxes.clear();
    scores.clear();
    classIds.clear();

    int validCount = 0;
    for (std::size_t i = 0; i < p.inferenceOutputs.size(); ++i)
    {
        auto &output = p.inferenceOutputs[i];
        int stride = static_cast<int>(p.modelWidth / output.gridWidth);

        validCount += processScale(
            output.data.data(),
            static_cast<int>(output.gridHeight),
            static_cast<int>(output.gridWidth),
            static_cast<int>(p.modelHeight),
            static_cast<int>(p.modelWidth),
            stride,
            boxes,
            scores,
            classIds);
    }

    if (validCount <= 0)
    {
        return;
    }

    auto &indices = p.sortedIndices;
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

    p.detections.clear();
    std::size_t detectionCount = 0;
    for (int i = 0; i < validCount && detectionCount < maxDetections_; ++i)
    {
        if (indices[static_cast<std::size_t>(i)] == -1)
        {
            continue;
        }

        int idx = indices[static_cast<std::size_t>(i)];
        float x1 = boxes[static_cast<std::size_t>(idx) * 4 + 0] - static_cast<float>(p.xPad);
        float y1 = boxes[static_cast<std::size_t>(idx) * 4 + 1] - static_cast<float>(p.yPad);
        float x2 = x1 + boxes[static_cast<std::size_t>(idx) * 4 + 2];
        float y2 = y1 + boxes[static_cast<std::size_t>(idx) * 4 + 3];

        YoloxDetectionResult detection;
        detection.left = static_cast<int>(clamp(x1, 0, static_cast<int>(p.modelWidth)) / p.scale);
        detection.top = static_cast<int>(clamp(y1, 0, static_cast<int>(p.modelHeight)) / p.scale);
        detection.right = static_cast<int>(clamp(x2, 0, static_cast<int>(p.modelWidth)) / p.scale);
        detection.bottom = static_cast<int>(clamp(y2, 0, static_cast<int>(p.modelHeight)) / p.scale);
        detection.classId = classIds[static_cast<std::size_t>(idx)];
        detection.confidence = scores[static_cast<std::size_t>(i)];

        p.detections.push_back(detection);
        ++detectionCount;
    }

    LOG.info("Frame idx=%d detections=%zu", p.idx, p.detections.size());
}

int PostprocessNode::processScale(float *input,
                                  int gridH,
                                  int gridW,
                                  int height,
                                  int width,
                                  int stride,
                                  std::vector<float> &boxes,
                                  std::vector<float> &scores,
                                  std::vector<int> &classIds) const
{
    (void)height;
    (void)width;
    int validCount = 0;
    int gridLen = gridH * gridW;

    for (int i = 0; i < gridH; ++i)
    {
        for (int j = 0; j < gridW; ++j)
        {
            float boxConfidence = input[4 * gridLen + i * gridW + j];
            if (boxConfidence < confThreshold_)
            {
                continue;
            }

            int offset = i * gridW + j;
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
            for (int k = 1; k < OBJ_CLASS_NUM; ++k)
            {
                float prob = inPtr[(5 + k) * gridLen];
                if (prob > maxClassProb)
                {
                    maxClassId = k;
                    maxClassProb = prob;
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

void PostprocessNode::quickSortIndicesInverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) const
{
    if (left >= right)
    {
        return;
    }

    float key = input[static_cast<std::size_t>(left)];
    int keyIndex = indices[static_cast<std::size_t>(left)];
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

float PostprocessNode::calculateIoU(float xmin0,
                                    float ymin0,
                                    float xmax0,
                                    float ymax0,
                                    float xmin1,
                                    float ymin1,
                                    float xmax1,
                                    float ymax1) const
{
    float w = std::max(0.0f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1) + 1.0f);
    float h = std::max(0.0f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1) + 1.0f);
    float intersect = w * h;
    float unionArea = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
                      (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - intersect;
    return unionArea <= 0.0f ? 0.0f : (intersect / unionArea);
}

void PostprocessNode::nms(int validCount,
                          const std::vector<float> &boxes,
                          const std::vector<int> &classIds,
                          std::vector<int> &order,
                          int filterId) const
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[static_cast<std::size_t>(i)];
        if (n == -1 || classIds[static_cast<std::size_t>(n)] != filterId)
        {
            continue;
        }

        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[static_cast<std::size_t>(j)];
            if (m == -1 || classIds[static_cast<std::size_t>(m)] != filterId)
            {
                continue;
            }

            float xmin0 = boxes[static_cast<std::size_t>(n) * 4 + 0];
            float ymin0 = boxes[static_cast<std::size_t>(n) * 4 + 1];
            float xmax0 = boxes[static_cast<std::size_t>(n) * 4 + 0] + boxes[static_cast<std::size_t>(n) * 4 + 2];
            float ymax0 = boxes[static_cast<std::size_t>(n) * 4 + 1] + boxes[static_cast<std::size_t>(n) * 4 + 3];

            float xmin1 = boxes[static_cast<std::size_t>(m) * 4 + 0];
            float ymin1 = boxes[static_cast<std::size_t>(m) * 4 + 1];
            float xmax1 = boxes[static_cast<std::size_t>(m) * 4 + 0] + boxes[static_cast<std::size_t>(m) * 4 + 2];
            float ymax1 = boxes[static_cast<std::size_t>(m) * 4 + 1] + boxes[static_cast<std::size_t>(m) * 4 + 3];

            float iou = calculateIoU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > nmsThreshold_)
            {
                order[static_cast<std::size_t>(j)] = -1;
            }
        }
    }
}

} // namespace YoloxNodes
