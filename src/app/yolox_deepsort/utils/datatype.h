#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <utility>
#include <vector>

inline constexpr int kFeatureDim = 512;

using FEATURE = Eigen::Matrix<float, 1, kFeatureDim, Eigen::RowMajor>;
using FEATURESS = Eigen::Matrix<float, Eigen::Dynamic, kFeatureDim, Eigen::RowMajor>;

using KAL_MEAN = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
using KAL_COVA = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;

using KAL_HMEAN = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using KAL_HCOVA = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

using DETECTBOX = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using DETECTBOXSS = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;

struct DETECTION_ROW
{
    DETECTBOX tlwh;
    float confidence;
    FEATURE feature;

    DETECTION_ROW(
        DETECTBOX t = DETECTBOX::Zero(),
        float c = 0.0f,
        FEATURE f = FEATURE::Zero())
        : tlwh(t),
          confidence(c),
          feature(f)
    {
    }

    DETECTBOX to_xyah() const
    {
        DETECTBOX ret = tlwh;
        ret(0) += ret(2) / 2.0f;
        ret(1) += ret(3) / 2.0f;
        ret(2) = ret(3) != 0.0f ? (ret(2) / ret(3)) : 0.0f;
        return ret;
    }
};

using DETECTIONS = std::vector<DETECTION_ROW>;
