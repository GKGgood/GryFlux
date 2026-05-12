#pragma once

#include "datatype.h"

#include <Eigen/Core>

#include <map>
#include <utility>
#include <vector>

using TRACKER_DATA = std::pair<int, FEATURESS>;
using DYNAMICM = Eigen::MatrixXf;

class NearNeighborDisMetric
{
public:
    enum METRIC_TYPE
    {
        euclidean,
        cosine
    };

    NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold, int budget);

    DYNAMICM distance(const FEATURESS &features, const std::vector<int> &targets);
    void partial_fit(std::vector<TRACKER_DATA> &tid_feats, std::vector<int> &active_targets);

    float mating_threshold;

private:
    using DistanceFunction =
        Eigen::VectorXf (NearNeighborDisMetric::*)(const FEATURESS &, const FEATURESS &);

    DistanceFunction metric_;
    int budget_;
    std::map<int, FEATURESS> samples_;

    Eigen::VectorXf nn_cosine_distance(const FEATURESS &x, const FEATURESS &y);
    Eigen::VectorXf nn_euclidean_distance(const FEATURESS &x, const FEATURESS &y);
    Eigen::MatrixXf pdist(const FEATURESS &x, const FEATURESS &y);
    Eigen::MatrixXf cosine_distance(
        const FEATURESS &a,
        const FEATURESS &b,
        bool data_is_normalized = false);
};
