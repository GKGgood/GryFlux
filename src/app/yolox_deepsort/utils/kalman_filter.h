#pragma once

#include "datatype.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <utility>
#include <vector>

class MyKalmanFilter
{
public:
    static const double chi2inv95[10];

    MyKalmanFilter();

    KAL_DATA initiate(const DETECTBOX &measurement);
    void predict(KAL_MEAN &mean, KAL_COVA &covariance);
    KAL_HDATA project(const KAL_MEAN &mean, const KAL_COVA &covariance);
    KAL_DATA update(
        const KAL_MEAN &mean,
        const KAL_COVA &covariance,
        const DETECTBOX &measurement);
    Eigen::Matrix<float, 1, -1> gating_distance(
        const KAL_MEAN &mean,
        const KAL_COVA &covariance,
        const std::vector<DETECTBOX> &measurements,
        bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8> motion_mat_;
    Eigen::Matrix<float, 4, 8> update_mat_;
    float std_weight_position_;
    float std_weight_velocity_;
};
