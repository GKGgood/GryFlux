#pragma once

#include "framework/node_base.h"

#include <cstddef>
#include <vector>

namespace PipelineNodes
{

class PostprocessNode : public GryFlux::NodeBase
{
public:
    PostprocessNode(
        int classCount,
        float confThreshold,
        float nmsThreshold,
        std::size_t maxDetections)
        : classCount_(classCount),
          confThreshold_(confThreshold),
          nmsThreshold_(nmsThreshold),
          maxDetections_(maxDetections)
    {
    }

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int processScale(
        float *input,
        int gridH,
        int gridW,
        int stride,
        std::vector<float> &boxes,
        std::vector<float> &scores,
        std::vector<int> &classIds) const;
    void quickSortIndicesInverse(
        std::vector<float> &input,
        int left,
        int right,
        std::vector<int> &indices) const;
    float calculateIoU(
        float xmin0,
        float ymin0,
        float xmax0,
        float ymax0,
        float xmin1,
        float ymin1,
        float xmax1,
        float ymax1) const;
    void nms(
        int validCount,
        const std::vector<float> &boxes,
        const std::vector<int> &classIds,
        std::vector<int> &order,
        int filterId) const;

    int classCount_;
    float confThreshold_;
    float nmsThreshold_;
    std::size_t maxDetections_;
};

} // namespace PipelineNodes
