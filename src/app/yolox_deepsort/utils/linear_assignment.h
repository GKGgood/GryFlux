#pragma once

#include "datatype.h"
#include "kalman_filter.h"
#include "track.h"

#include <Eigen/Core>
#include <vector>

using DYNAMICM = Eigen::MatrixXf;
using MATCH_DATA = std::pair<int, int>;

struct TRACKER_MATCHD
{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

class DeepSortTracker;

class linear_assignment
{
public:
    using GATED_METRIC_FUNC = DYNAMICM (DeepSortTracker::*)(
        std::vector<Track> &,
        const DETECTIONS &,
        const std::vector<int> &,
        const std::vector<int> &);

    static linear_assignment *getInstance();

    TRACKER_MATCHD matching_cascade(
        DeepSortTracker *distance_metric,
        GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        int cascade_depth,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int> &track_indices,
        std::vector<int> detection_indices = {});

    TRACKER_MATCHD min_cost_matching(
        DeepSortTracker *distance_metric,
        GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int> &track_indices,
        std::vector<int> &detection_indices);

    DYNAMICM gate_cost_matrix(
        MyKalmanFilter *kf,
        DYNAMICM &cost_matrix,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices,
        float gated_cost = 10000.0f,
        bool only_position = false);

private:
    linear_assignment();
    static linear_assignment *instance_;
};

const float INFTY_COST = 10000.0f;
