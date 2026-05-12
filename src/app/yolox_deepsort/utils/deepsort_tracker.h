#pragma once

#include "datatype.h"
#include "kalman_filter.h"
#include "nn_matching.h"
#include "track.h"

#include <vector>

class DeepSortTracker
{
public:
    DeepSortTracker(
        float max_cosine_distance = 0.5f,
        int nn_budget = 100,
        int max_age = 30,
        int n_init = 3);
    ~DeepSortTracker() = default;

    std::vector<Track> update(const DETECTIONS &detections);

    DYNAMICM gated_metric(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices);
    DYNAMICM iou_cost(
        std::vector<Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices);

private:
    NearNeighborDisMetric metric_;
    MyKalmanFilter kf_;
    std::vector<Track> tracks_;
    int next_id_;

    float max_cosine_distance_;
    int nn_budget_;
    int max_age_;
    int n_init_;
};
