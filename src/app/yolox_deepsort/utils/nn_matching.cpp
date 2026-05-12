#include "nn_matching.h"

#include <cmath>
#include <limits>

NearNeighborDisMetric::NearNeighborDisMetric(
    NearNeighborDisMetric::METRIC_TYPE metric,
    float matching_threshold,
    int budget)
{
    if (metric == euclidean)
    {
        metric_ = &NearNeighborDisMetric::nn_euclidean_distance;
    }
    else
    {
        metric_ = &NearNeighborDisMetric::nn_cosine_distance;
    }

    mating_threshold = matching_threshold;
    budget_ = budget;
    samples_.clear();
}

DYNAMICM NearNeighborDisMetric::distance(
    const FEATURESS &features,
    const std::vector<int> &targets)
{
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
    int idx = 0;
    for (int target : targets)
    {
        cost_matrix.row(idx) = (this->*metric_)(samples_[target], features);
        ++idx;
    }
    return cost_matrix;
}

void NearNeighborDisMetric::partial_fit(
    std::vector<TRACKER_DATA> &tid_feats,
    std::vector<int> &active_targets)
{
    for (TRACKER_DATA &data : tid_feats)
    {
        const int track_id = data.first;
        const FEATURESS newFeatOne = data.second;

        if (samples_.find(track_id) != samples_.end())
        {
            const int oldSize = samples_[track_id].rows();
            const int addSize = newFeatOne.rows();
            const int newSize = oldSize + addSize;

            if (newSize <= budget_)
            {
                FEATURESS newSampleFeatures(newSize, kFeatureDim);
                newSampleFeatures.block(0, 0, oldSize, kFeatureDim) = samples_[track_id];
                newSampleFeatures.block(oldSize, 0, addSize, kFeatureDim) = newFeatOne;
                samples_[track_id] = newSampleFeatures;
            }
            else
            {
                if (oldSize < budget_)
                {
                    FEATURESS newSampleFeatures(budget_, kFeatureDim);
                    if (addSize >= budget_)
                    {
                        newSampleFeatures = newFeatOne.block(0, 0, budget_, kFeatureDim);
                    }
                    else
                    {
                        newSampleFeatures.block(0, 0, budget_ - addSize, kFeatureDim) =
                            samples_[track_id]
                                .block(
                                    oldSize - (budget_ - addSize),
                                    0,
                                    budget_ - addSize,
                                    kFeatureDim)
                                .eval();
                        newSampleFeatures.block(
                            budget_ - addSize,
                            0,
                            addSize,
                            kFeatureDim) = newFeatOne;
                    }
                    samples_[track_id] = newSampleFeatures;
                }
                else
                {
                    if (addSize >= budget_)
                    {
                        samples_[track_id] = newFeatOne.block(0, 0, budget_, kFeatureDim);
                    }
                    else
                    {
                        samples_[track_id].block(0, 0, budget_ - addSize, kFeatureDim) =
                            samples_[track_id]
                                .block(
                                    oldSize - (budget_ - addSize),
                                    0,
                                    budget_ - addSize,
                                    kFeatureDim)
                                .eval();
                        samples_[track_id].block(
                            budget_ - addSize,
                            0,
                            addSize,
                            kFeatureDim) = newFeatOne;
                    }
                }
            }
        }
        else
        {
            samples_[track_id] = newFeatOne;
        }
    }

    for (auto it = samples_.begin(); it != samples_.end();)
    {
        bool active = false;
        for (int target : active_targets)
        {
            if (target == it->first)
            {
                active = true;
                break;
            }
        }
        if (!active)
        {
            samples_.erase(it++);
        }
        else
        {
            ++it;
        }
    }
}

Eigen::VectorXf NearNeighborDisMetric::nn_cosine_distance(
    const FEATURESS &x,
    const FEATURESS &y)
{
    if (x.rows() == 0 || y.rows() == 0)
    {
        return Eigen::VectorXf::Constant(y.rows(), 2.0f);
    }
    Eigen::MatrixXf distances = cosine_distance(x, y);
    return distances.colwise().minCoeff().transpose();
}

Eigen::VectorXf NearNeighborDisMetric::nn_euclidean_distance(
    const FEATURESS &x,
    const FEATURESS &y)
{
    if (x.rows() == 0 || y.rows() == 0)
    {
        return Eigen::VectorXf::Constant(y.rows(), std::numeric_limits<float>::max());
    }
    Eigen::MatrixXf distances = pdist(x, y);
    Eigen::VectorXf res = distances.colwise().maxCoeff().transpose();
    res = res.array().max(Eigen::VectorXf::Zero(res.rows()).array());
    return res;
}

Eigen::MatrixXf NearNeighborDisMetric::pdist(
    const FEATURESS &x,
    const FEATURESS &y)
{
    const int len1 = x.rows();
    const int len2 = y.rows();
    if (len1 == 0 || len2 == 0)
    {
        return Eigen::MatrixXf::Zero(len1, len2);
    }

    Eigen::MatrixXf res = -2.0f * x * y.transpose();
    res = res.colwise() + x.rowwise().squaredNorm();
    res = res.rowwise() + y.rowwise().squaredNorm().transpose();
    res = res.array().max(Eigen::MatrixXf::Zero(res.rows(), res.cols()).array());
    return res;
}

Eigen::MatrixXf NearNeighborDisMetric::cosine_distance(
    const FEATURESS &a,
    const FEATURESS &b,
    bool data_is_normalized)
{
    FEATURESS aa = a;
    FEATURESS bb = b;
    if (!data_is_normalized)
    {
        for (int i = 0; i < a.rows(); ++i)
        {
            const float norm = std::sqrt(a.row(i).squaredNorm());
            aa.row(i) = a.row(i) / norm;
        }
        for (int i = 0; i < b.rows(); ++i)
        {
            const float norm = std::sqrt(b.row(i).squaredNorm());
            bb.row(i) = b.row(i) / norm;
        }
    }

    return 1.0f - (aa * bb.transpose()).array();
}
