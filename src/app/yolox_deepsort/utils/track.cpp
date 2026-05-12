#include "track.h"

#include "kalman_filter.h"

Track::Track(
    KAL_MEAN mean,
    KAL_COVA covariance,
    int track_id,
    int n_init,
    int max_age,
    const FEATURE &feature)
    : mean(mean),
      covariance(covariance),
      track_id(track_id),
      hits(1),
      age(1),
      time_since_update(0),
      state(TrackState::Tentative),
      n_init_(n_init),
      max_age_(max_age)
{
    features = feature;
}

void Track::predict(MyKalmanFilter *kf)
{
    kf->predict(mean, covariance);
    ++age;
    ++time_since_update;
}

void Track::update(MyKalmanFilter *kf, const DETECTION_ROW &detection)
{
    KAL_DATA result = kf->update(mean, covariance, detection.to_xyah());
    mean = result.first;
    covariance = result.second;
    features = detection.feature;

    ++hits;
    time_since_update = 0;
    if (state == TrackState::Tentative && hits >= n_init_)
    {
        state = TrackState::Confirmed;
    }
}

DETECTBOX Track::to_tlwh() const
{
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret(0) -= ret(2) / 2.0f;
    ret(1) -= ret(3) / 2.0f;
    return ret;
}
