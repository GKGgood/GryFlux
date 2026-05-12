#pragma once

#include "framework/data_packet.h"
#include "packet/rknn_tensor.h"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct YoloxDetectionResult
{
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
    int classId = -1;
    float confidence = 0.0f;
};

struct YoloxPacket : public GryFlux::DataPacket
{
    int idx = 0;
    std::string imagePath;

    cv::Mat originalImage;
    std::vector<std::uint8_t> inputTensor;

    std::size_t modelWidth = 640;
    std::size_t modelHeight = 640;
    float scale = 1.0f;
    int xPad = 0;
    int yPad = 0;

    std::vector<GryFlux::RknnTensor> inferenceOutputs;
    std::vector<YoloxDetectionResult> detections;

    // Scratch buffers reused by postprocess to reduce per-frame allocations.
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    std::vector<int> sortedIndices;

    YoloxPacket()
    {
        constexpr std::size_t kMaxCandidates = 80 * 80 + 40 * 40 + 20 * 20;
        detections.reserve(128);
        boxes.reserve(kMaxCandidates * 4);
        scores.reserve(kMaxCandidates);
        classIds.reserve(kMaxCandidates);
        sortedIndices.reserve(kMaxCandidates);
    }

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(idx);
    }
};
