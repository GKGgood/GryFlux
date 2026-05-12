#pragma once

#include "framework/data_packet.h"
#include "packet/rknn_tensor.h"
#include "utils/track.h"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Detection
{
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = -1;
};

struct TrackDataPacket : public GryFlux::DataPacket
{
    int frame_id = 0;
    cv::Mat original_image;

    std::vector<std::uint8_t> detection_input_tensor;
    std::size_t detection_model_width = 0;
    std::size_t detection_model_height = 0;
    float detection_scale = 1.0f;
    int detection_x_pad = 0;
    int detection_y_pad = 0;

    std::vector<GryFlux::RknnTensor> detection_outputs;
    std::vector<Detection> detections;

    std::vector<std::vector<std::uint8_t>> reid_input_tensors;
    std::vector<std::uint8_t> reid_input_valid_flags;
    std::size_t active_reid_input_count = 0;

    std::vector<std::vector<float>> reid_features;
    std::size_t active_reid_feature_count = 0;

    std::vector<Track> active_tracks;
    std::size_t max_detection_capacity = 0;

    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<int> sorted_indices;

    TrackDataPacket(
        std::size_t detection_capacity,
        int reid_width,
        int reid_height,
        int reid_feature_dim)
        : max_detection_capacity(detection_capacity)
    {
        detections.reserve(detection_capacity);
        reid_input_tensors.resize(detection_capacity);
        reid_input_valid_flags.resize(detection_capacity, 0U);
        reid_features.resize(detection_capacity);
        active_tracks.reserve(detection_capacity);

        const std::size_t reid_input_size =
            static_cast<std::size_t>(reid_width) *
            static_cast<std::size_t>(reid_height) * 3U;
        const std::size_t reid_feature_size =
            static_cast<std::size_t>(reid_feature_dim > 0 ? reid_feature_dim : 0);

        for (std::size_t i = 0; i < detection_capacity; ++i)
        {
            reid_input_tensors[i].resize(reid_input_size);
            reid_features[i].resize(reid_feature_size, 0.0f);
        }

        constexpr std::size_t kMaxCandidates = 80U * 80U + 40U * 40U + 20U * 20U;
        boxes.reserve(kMaxCandidates * 4U);
        scores.reserve(kMaxCandidates);
        class_ids.reserve(kMaxCandidates);
        sorted_indices.reserve(kMaxCandidates);
    }

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(frame_id);
    }
};
