#include "kalman_filter.h"

#include <Eigen/Cholesky>

#include <iostream>

const double MyKalmanFilter::chi2inv95[10] = {
    0,
    3.8415,
    5.9915,
    7.8147,
    9.4877,
    11.070,
    12.592,
    14.067,
    15.507,
    16.919};

MyKalmanFilter::MyKalmanFilter()
{
    constexpr int ndim = 4;
    constexpr double dt = 1.0;

    motion_mat_ = Eigen::Matrix<float, 8, 8>::Identity();
    for (int i = 0; i < ndim; ++i)
    {
        motion_mat_(i, ndim + i) = static_cast<float>(dt);
    }

    update_mat_ = Eigen::Matrix<float, 4, 8>::Identity();
    std_weight_position_ = 1.0f / 20.0f;
    std_weight_velocity_ = 1.0f / 160.0f;
}

KAL_DATA MyKalmanFilter::initiate(const DETECTBOX &measurement)
{
    DETECTBOX meanPos = measurement;
    DETECTBOX meanVel = DETECTBOX::Zero();

    KAL_MEAN mean;
    mean.block<1, 4>(0, 0) = meanPos;
    mean.block<1, 4>(0, 4) = meanVel;

    KAL_MEAN stdDev;
    stdDev(0) = 2.0f * std_weight_position_ * measurement(3);
    stdDev(1) = 2.0f * std_weight_position_ * measurement(3);
    stdDev(2) = 1e-2f;
    stdDev(3) = 2.0f * std_weight_position_ * measurement(3);
    stdDev(4) = 10.0f * std_weight_velocity_ * measurement(3);
    stdDev(5) = 10.0f * std_weight_velocity_ * measurement(3);
    stdDev(6) = 1e-5f;
    stdDev(7) = 10.0f * std_weight_velocity_ * measurement(3);

    KAL_COVA covariance = stdDev.array().square().matrix().asDiagonal();
    return std::make_pair(mean, covariance);
}

void MyKalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
{
    DETECTBOX stdPos;
    stdPos << std_weight_position_ * mean(3),
        std_weight_position_ * mean(3),
        1e-2f,
        std_weight_position_ * mean(3);
    DETECTBOX stdVel;
    stdVel << std_weight_velocity_ * mean(3),
        std_weight_velocity_ * mean(3),
        1e-5f,
        std_weight_velocity_ * mean(3);

    KAL_MEAN stdDevCombined;
    stdDevCombined.block<1, 4>(0, 0) = stdPos;
    stdDevCombined.block<1, 4>(0, 4) = stdVel;

    KAL_COVA motionCov = stdDevCombined.array().square().matrix().asDiagonal();

    mean = (motion_mat_ * mean.transpose()).transpose();
    covariance = motion_mat_ * covariance * motion_mat_.transpose();
    covariance += motionCov;
}

KAL_HDATA MyKalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance)
{
    DETECTBOX stdDev;
    stdDev << std_weight_position_ * mean(3),
        std_weight_position_ * mean(3),
        1e-1f,
        std_weight_position_ * mean(3);

    KAL_HCOVA innovationCov = stdDev.array().square().matrix().asDiagonal();
    KAL_HMEAN projectedMean = (update_mat_ * mean.transpose()).transpose();
    KAL_HCOVA projectedCov = update_mat_ * covariance * update_mat_.transpose();
    projectedCov += innovationCov;

    return std::make_pair(projectedMean, projectedCov);
}

KAL_DATA MyKalmanFilter::update(
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const DETECTBOX &measurement)
{
    const KAL_HDATA projected = project(mean, covariance);
    const KAL_HMEAN projectedMean = projected.first;
    const KAL_HCOVA projectedCov = projected.second;

    Eigen::Matrix<float, 4, 8> b = update_mat_ * covariance;
    Eigen::Matrix<float, 8, 4> kalmanGain =
        (projectedCov.llt().solve(b)).transpose();

    Eigen::Matrix<float, 1, 4> innovation = measurement - projectedMean;

    KAL_MEAN newMean = mean + innovation * kalmanGain.transpose();
    KAL_COVA newCovariance =
        covariance - kalmanGain * update_mat_ * covariance;

    return std::make_pair(newMean, newCovariance);
}

Eigen::Matrix<float, 1, -1> MyKalmanFilter::gating_distance(
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const std::vector<DETECTBOX> &measurements,
    bool only_position)
{
    const KAL_HDATA projected = project(mean, covariance);
    const KAL_HMEAN projectedMean = projected.first;
    const KAL_HCOVA projectedCov = projected.second;

    if (only_position)
    {
        std::cerr << "gating_distance with only_position=true is not implemented!"
                  << std::endl;
        return Eigen::Matrix<float, 1, -1>::Constant(1, measurements.size(), 1e5f);
    }

    DETECTBOXSS diff(measurements.size(), 4);
    for (std::size_t i = 0; i < measurements.size(); ++i)
    {
        diff.row(static_cast<int>(i)) = measurements[i] - projectedMean;
    }

    Eigen::Matrix<float, 4, 4> l = projectedCov.llt().matrixL();
    Eigen::Matrix<float, 4, -1> y =
        l.triangularView<Eigen::Lower>().solve(diff.transpose());

    return y.array().square().colwise().sum();
}
