#include "deepsort_tracker.h"

#include "linear_assignment.h"

#include <algorithm>

DeepSortTracker::DeepSortTracker(
    float max_cosine_distance,
    int nn_budget,
    int max_age,
    int n_init)
    : metric_(NearNeighborDisMetric::cosine, max_cosine_distance, nn_budget),
      next_id_(1),
      max_cosine_distance_(max_cosine_distance),
      nn_budget_(nn_budget),
      max_age_(max_age),
      n_init_(n_init)
{
}

std::vector<Track> DeepSortTracker::update(const DETECTIONS &detections)
{
    for (auto &track : tracks_)
    {
        track.predict(&kf_);
    }

    std::vector<int> confirmedTracks;
    std::vector<int> unconfirmedTracks;
    for (std::size_t i = 0; i < tracks_.size(); ++i)
    {
        if (tracks_[i].is_confirmed())
        {
            confirmedTracks.push_back(static_cast<int>(i));
        }
        else
        {
            unconfirmedTracks.push_back(static_cast<int>(i));
        }
    }

    std::vector<int> detectionIndices;
    for (std::size_t i = 0; i < detections.size(); ++i)
    {
        detectionIndices.push_back(static_cast<int>(i));
    }

    TRACKER_MATCHD resA = linear_assignment::getInstance()->matching_cascade(
        this,
        &DeepSortTracker::gated_metric,
        max_cosine_distance_,
        max_age_,
        tracks_,
        detections,
        confirmedTracks,
        detectionIndices);

    std::vector<std::pair<int, int>> matchesA = resA.matches;
    std::vector<int> unmatchedTracksA = resA.unmatched_tracks;
    std::vector<int> unmatchedDetections = resA.unmatched_detections;

    std::vector<int> iouTrackCandidates;
    for (int t : unmatchedTracksA)
    {
        if (tracks_[static_cast<std::size_t>(t)].time_since_update == 1)
        {
            iouTrackCandidates.push_back(t);
        }
    }
    for (int t : unconfirmedTracks)
    {
        iouTrackCandidates.push_back(t);
    }

    TRACKER_MATCHD resB = linear_assignment::getInstance()->min_cost_matching(
        this,
        &DeepSortTracker::iou_cost,
        0.7f,
        tracks_,
        detections,
        iouTrackCandidates,
        unmatchedDetections);

    std::vector<std::pair<int, int>> matches = matchesA;
    matches.insert(matches.end(), resB.matches.begin(), resB.matches.end());

    for (const auto &match : matches)
    {
        tracks_[static_cast<std::size_t>(match.first)].update(
            &kf_,
            detections[static_cast<std::size_t>(match.second)]);
    }

    std::vector<int> allUnmatchedTracks;
    for (int t : unmatchedTracksA)
    {
        if (tracks_[static_cast<std::size_t>(t)].time_since_update > 1)
        {
            allUnmatchedTracks.push_back(t);
        }
    }
    allUnmatchedTracks.insert(
        allUnmatchedTracks.end(),
        resB.unmatched_tracks.begin(),
        resB.unmatched_tracks.end());

    for (int trackIdx : allUnmatchedTracks)
    {
        Track &track = tracks_[static_cast<std::size_t>(trackIdx)];
        if (track.state == TrackState::Tentative || track.time_since_update > max_age_)
        {
            track.state = TrackState::Deleted;
        }
    }

    for (int detIdx : resB.unmatched_detections)
    {
        KAL_DATA data = kf_.initiate(
            detections[static_cast<std::size_t>(detIdx)].to_xyah());
        tracks_.emplace_back(
            data.first,
            data.second,
            next_id_++,
            n_init_,
            max_age_,
            detections[static_cast<std::size_t>(detIdx)].feature);
    }

    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [](const Track &t) { return t.is_deleted(); }),
        tracks_.end());

    std::vector<int> activeTargets;
    std::vector<TRACKER_DATA> tidFeatures;
    for (const auto &track : tracks_)
    {
        if (!track.is_confirmed())
        {
            continue;
        }
        activeTargets.push_back(track.track_id);
        tidFeatures.push_back({track.track_id, track.features});
    }
    metric_.partial_fit(tidFeatures, activeTargets);

    std::vector<Track> activeTracks;
    for (const auto &track : tracks_)
    {
        if (track.is_confirmed() && track.time_since_update <= 1)
        {
            activeTracks.push_back(track);
        }
    }
    return activeTracks;
}

DYNAMICM DeepSortTracker::gated_metric(
    std::vector<Track> &tracks,
    const DETECTIONS &detections,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices)
{
    FEATURESS features(detection_indices.size(), kFeatureDim);
    for (std::size_t i = 0; i < detection_indices.size(); ++i)
    {
        features.row(static_cast<int>(i)) =
            detections[static_cast<std::size_t>(detection_indices[i])].feature;
    }

    std::vector<int> targets;
    for (int trackIndex : track_indices)
    {
        targets.push_back(tracks[static_cast<std::size_t>(trackIndex)].track_id);
    }

    DYNAMICM cost_matrix = metric_.distance(features, targets);

    std::vector<DETECTBOX> measurements;
    for (int dIdx : detection_indices)
    {
        measurements.push_back(detections[static_cast<std::size_t>(dIdx)].to_xyah());
    }

    for (std::size_t i = 0; i < track_indices.size(); ++i)
    {
        auto &track = tracks[static_cast<std::size_t>(track_indices[i])];
        Eigen::Matrix<float, 1, -1> gate_dists =
            kf_.gating_distance(track.mean, track.covariance, measurements, false);

        for (std::size_t j = 0; j < detection_indices.size(); ++j)
        {
            if (gate_dists(0, static_cast<int>(j)) > 9.4877f)
            {
                cost_matrix(
                    static_cast<int>(i),
                    static_cast<int>(j)) = 100000.0f;
            }
        }
    }

    return cost_matrix;
}

DYNAMICM DeepSortTracker::iou_cost(
    std::vector<Track> &tracks,
    const DETECTIONS &detections,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices)
{
    DYNAMICM cost_matrix =
        Eigen::MatrixXf::Zero(track_indices.size(), detection_indices.size());

    for (std::size_t i = 0; i < track_indices.size(); ++i)
    {
        const auto track_box =
            tracks[static_cast<std::size_t>(track_indices[i])].to_tlwh();
        for (std::size_t j = 0; j < detection_indices.size(); ++j)
        {
            const auto det_box =
                detections[static_cast<std::size_t>(detection_indices[j])].tlwh;

            const float ix = std::max(track_box(0), det_box(0));
            const float iy = std::max(track_box(1), det_box(1));
            const float iw =
                std::min(track_box(0) + track_box(2), det_box(0) + det_box(2)) - ix;
            const float ih =
                std::min(track_box(1) + track_box(3), det_box(1) + det_box(3)) - iy;

            float iou = 0.0f;
            if (iw > 0.0f && ih > 0.0f)
            {
                const float intersection = iw * ih;
                const float union_area =
                    track_box(2) * track_box(3) + det_box(2) * det_box(3) -
                    intersection;
                iou = intersection / union_area;
            }

            cost_matrix(
                static_cast<int>(i),
                static_cast<int>(j)) = 1.0f - iou;
        }
    }
    return cost_matrix;
}
