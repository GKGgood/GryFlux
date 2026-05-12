#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct ZeroDcePacket : public GryFlux::DataPacket
{
    explicit ZeroDcePacket(int inputWidthArg, int inputHeightArg)
        : input_width(inputWidthArg),
          input_height(inputHeightArg)
    {
    }

    uint64_t getIdx() const override
    {
        return frame_id;
    }

    uint64_t frame_id = 0;
    std::string output_relative_path;

    int input_width = 0;
    int input_height = 0;
    int output_channels = 0;
    int output_width = 0;
    int output_height = 0;

    cv::Mat input_image;
    cv::Mat output_image;

    std::vector<std::uint8_t> input_tensor;
    std::vector<float> output_tensor;

    bool is_valid_image = true;
};
