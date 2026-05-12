#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct DeeplabPacket : public GryFlux::DataPacket
{
    struct InferenceOutput
    {
        std::vector<float> data;
        std::size_t gridH = 0;
        std::size_t gridW = 0;
        std::size_t channels = 0;
    };

    int idx = 0;
    std::string imagePath;

    cv::Mat originalImage;
    cv::Mat preprocessedImage;
    std::vector<uint8_t> inputData;

    std::size_t modelWidth = 0;
    std::size_t modelHeight = 0;
    std::size_t resizedWidth = 0;
    std::size_t resizedHeight = 0;
    float scale = 1.0f;
    int xPad = 0;
    int yPad = 0;

    std::vector<InferenceOutput> inferenceOutputs;
    cv::Mat mask;

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(idx);
    }
};
