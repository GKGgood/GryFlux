#pragma once

#include "datatype.h"

enum class TrackState
{
    Tentative = 1,
    Confirmed = 2,
    Deleted = 3
};

class Track
{
public:
    KAL_MEAN mean;
    KAL_COVA covariance;

    int track_id;
    int hits;
    int age;
    int time_since_update;
    TrackState state;

    FEATURESS features;

    Track(
        KAL_MEAN mean,
        KAL_COVA covariance,
        int track_id,
        int n_init,
        int max_age,
        const FEATURE &feature);

    void predict(class MyKalmanFilter *kf);
    void update(class MyKalmanFilter *kf, const DETECTION_ROW &detection);
    DETECTBOX to_tlwh() const;

    bool is_confirmed() const { return state == TrackState::Confirmed; }
    bool is_deleted() const { return state == TrackState::Deleted; }

private:
    int n_init_;
    int max_age_;
};
