#include "linear_assignment.h"

#include "deepsort_tracker.h"
#include "munkres.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

namespace
{

Eigen::Matrix<float, -1, 2, Eigen::RowMajor> solveHungarian(const DYNAMICM &cost_matrix)
{
    const int rows = static_cast<int>(cost_matrix.rows());
    const int cols = static_cast<int>(cost_matrix.cols());

    if (rows == 0 || cols == 0)
    {
        return Eigen::Matrix<float, -1, 2, Eigen::RowMajor>(0, 2);
    }

    Matrix<double> matrix(rows, cols);
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            if (std::isnan(cost_matrix(row, col)) || std::isinf(cost_matrix(row, col)))
            {
                matrix(row, col) = std::numeric_limits<double>::max() / 2.0;
            }
            else
            {
                matrix(row, col) = static_cast<double>(cost_matrix(row, col));
            }
        }
    }

    Munkres<double> solver;
    solver.solve(matrix);

    std::vector<std::pair<int, int>> pairs;
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            if (matrix(row, col) == 0)
            {
                pairs.emplace_back(row, col);
            }
        }
    }

    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> result(
        static_cast<int>(pairs.size()),
        2);
    for (int i = 0; i < result.rows(); ++i)
    {
        result(i, 0) = static_cast<float>(pairs[static_cast<std::size_t>(i)].first);
        result(i, 1) = static_cast<float>(pairs[static_cast<std::size_t>(i)].second);
    }
    return result;
}

} // namespace

linear_assignment *linear_assignment::instance_ = nullptr;

linear_assignment::linear_assignment() = default;

linear_assignment *linear_assignment::getInstance()
{
    if (instance_ == nullptr)
    {
        instance_ = new linear_assignment();
    }
    return instance_;
}

TRACKER_MATCHD linear_assignment::matching_cascade(
    DeepSortTracker *distance_metric,
    GATED_METRIC_FUNC distance_metric_func,
    float max_distance,
    int cascade_depth,
    std::vector<Track> &tracks,
    const DETECTIONS &detections,
    std::vector<int> &track_indices,
    std::vector<int> detection_indices)
{
    TRACKER_MATCHD result;

    if (detection_indices.empty() && !detections.empty())
    {
        detection_indices.resize(detections.size());
        for (std::size_t i = 0; i < detections.size(); ++i)
        {
            detection_indices[i] = static_cast<int>(i);
        }
    }

    std::vector<int> unmatchedDetections = detection_indices;
    std::map<int, int> matchedTracks;

    for (int level = 0; level < cascade_depth; ++level)
    {
        if (unmatchedDetections.empty())
        {
            break;
        }

        std::vector<int> levelTrackIndices;
        for (int trackIndex : track_indices)
        {
            if (trackIndex < 0 || static_cast<std::size_t>(trackIndex) >= tracks.size())
            {
                continue;
            }
            if (tracks[static_cast<std::size_t>(trackIndex)].time_since_update == level + 1)
            {
                levelTrackIndices.push_back(trackIndex);
            }
        }

        if (levelTrackIndices.empty())
        {
            continue;
        }

        TRACKER_MATCHD temp = min_cost_matching(
            distance_metric,
            distance_metric_func,
            max_distance,
            tracks,
            detections,
            levelTrackIndices,
            unmatchedDetections);

        unmatchedDetections = temp.unmatched_detections;
        for (const auto &match : temp.matches)
        {
            result.matches.push_back(match);
            matchedTracks[match.first] = match.second;
        }
    }

    result.unmatched_detections = unmatchedDetections;

    for (int trackIndex : track_indices)
    {
        if (trackIndex < 0 || static_cast<std::size_t>(trackIndex) >= tracks.size())
        {
            continue;
        }
        if (matchedTracks.find(trackIndex) == matchedTracks.end())
        {
            result.unmatched_tracks.push_back(trackIndex);
        }
    }

    return result;
}

TRACKER_MATCHD linear_assignment::min_cost_matching(
    DeepSortTracker *distance_metric,
    GATED_METRIC_FUNC distance_metric_func,
    float max_distance,
    std::vector<Track> &tracks,
    const DETECTIONS &detections,
    std::vector<int> &track_indices,
    std::vector<int> &detection_indices)
{
    TRACKER_MATCHD result;

    if (detection_indices.empty() || track_indices.empty())
    {
        result.unmatched_tracks = track_indices;
        result.unmatched_detections = detection_indices;
        return result;
    }

    DYNAMICM cost_matrix = (distance_metric->*distance_metric_func)(
        tracks,
        detections,
        track_indices,
        detection_indices);

    for (int i = 0; i < cost_matrix.rows(); ++i)
    {
        for (int j = 0; j < cost_matrix.cols(); ++j)
        {
            const float cost = cost_matrix(i, j);
            if (std::isnan(cost) || std::isinf(cost) || cost > max_distance)
            {
                cost_matrix(i, j) = INFTY_COST;
            }
        }
    }

    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = solveHungarian(cost_matrix);

    std::vector<bool> trackMatched(track_indices.size(), false);
    std::vector<bool> detectionMatched(detection_indices.size(), false);

    for (int i = 0; i < indices.rows(); ++i)
    {
        const int row = static_cast<int>(indices(i, 0));
        const int col = static_cast<int>(indices(i, 1));

        if (row < 0 || static_cast<std::size_t>(row) >= track_indices.size() ||
            col < 0 || static_cast<std::size_t>(col) >= detection_indices.size())
        {
            continue;
        }

        if (row < cost_matrix.rows() && col < cost_matrix.cols() &&
            cost_matrix(row, col) < max_distance)
        {
            result.matches.emplace_back(track_indices[static_cast<std::size_t>(row)],
                                        detection_indices[static_cast<std::size_t>(col)]);
            trackMatched[static_cast<std::size_t>(row)] = true;
            detectionMatched[static_cast<std::size_t>(col)] = true;
        }
    }

    for (std::size_t i = 0; i < track_indices.size(); ++i)
    {
        if (!trackMatched[i])
        {
            result.unmatched_tracks.push_back(track_indices[i]);
        }
    }

    for (std::size_t i = 0; i < detection_indices.size(); ++i)
    {
        if (!detectionMatched[i])
        {
            result.unmatched_detections.push_back(detection_indices[i]);
        }
    }

    return result;
}

DYNAMICM linear_assignment::gate_cost_matrix(
    MyKalmanFilter *kf,
    DYNAMICM &cost_matrix,
    std::vector<Track> &tracks,
    const DETECTIONS &detections,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices,
    float gated_cost,
    bool only_position)
{
    const int gating_dim = only_position ? 2 : 4;
    if (gating_dim <= 0 || gating_dim >= 10)
    {
        std::cerr << "Invalid gating_dim in gate_cost_matrix: " << gating_dim << std::endl;
        return cost_matrix;
    }
    const double gating_threshold = MyKalmanFilter::chi2inv95[gating_dim];

    std::vector<DETECTBOX> measurements;
    measurements.reserve(detection_indices.size());
    for (int idx : detection_indices)
    {
        if (idx < 0 || static_cast<std::size_t>(idx) >= detections.size())
        {
            continue;
        }
        measurements.push_back(detections[static_cast<std::size_t>(idx)].to_xyah());
    }

    if (measurements.empty() && !detection_indices.empty())
    {
        cost_matrix.fill(gated_cost);
        return cost_matrix;
    }
    if (measurements.empty())
    {
        return cost_matrix;
    }

    for (std::size_t i = 0; i < track_indices.size(); ++i)
    {
        const int trackIdx = track_indices[i];
        if (trackIdx < 0 || static_cast<std::size_t>(trackIdx) >= tracks.size())
        {
            continue;
        }

        Track &track = tracks[static_cast<std::size_t>(trackIdx)];
        Eigen::Matrix<float, 1, -1> gating_distance = kf->gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position);

        int validMeasurementIdx = 0;
        for (std::size_t j = 0; j < detection_indices.size(); ++j)
        {
            const int detIdx = detection_indices[j];
            if (detIdx < 0 || static_cast<std::size_t>(detIdx) >= detections.size())
            {
                continue;
            }

            if (validMeasurementIdx >= gating_distance.cols())
            {
                break;
            }

            if (gating_distance(0, validMeasurementIdx) > gating_threshold)
            {
                cost_matrix(static_cast<int>(i), static_cast<int>(j)) =
                    gated_cost;
            }
            ++validMeasurementIdx;
        }
    }

    return cost_matrix;
}
