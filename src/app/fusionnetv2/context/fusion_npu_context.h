#pragma once

#include "framework/context.h"

#include <opencv2/core.hpp>
#include <rknn_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class FusionNpuContext : public GryFlux::Context
{
public:
    explicit FusionNpuContext(int deviceId,
                              const std::string &modelPath,
                              int expectedModelWidth = 0,
                              int expectedModelHeight = 0);
    ~FusionNpuContext() override;

    int getModelWidth() const { return modelWidth_; }
    int getModelHeight() const { return modelHeight_; }
    std::size_t getOutputCount() const { return outputAttrs_.size(); }

    void setInputs(const cv::Mat &visibleYF32, const cv::Mat &infraredF32);
    void runInference();
    cv::Mat getOutput(std::size_t index);

private:
    static rknn_core_mask toCoreMask(int deviceId);
    static float inputScaleForAttr(const rknn_tensor_attr &attr);
    static std::size_t tensorTypeSize(rknn_tensor_type type);
    static void resolveSpatial(const rknn_tensor_attr &attr, int &height, int &width);
    static void dumpTensorAttr(const rknn_tensor_attr &attr);

    void loadModel(const std::string &path);
    void prepareInputTensors();
    void prepareOutputTensors();
    void releaseOutputs();
    void releaseResources();
    void validateInput(const cv::Mat &mat, const char *name) const;
    void setInput(std::size_t index, const cv::Mat &mat);

    int deviceId_ = 0;
    int expectedModelWidth_ = 0;
    int expectedModelHeight_ = 0;
    int modelWidth_ = 0;
    int modelHeight_ = 0;

    rknn_context ctx_ = 0;
    std::vector<std::uint8_t> modelData_;
    std::vector<rknn_tensor_attr> inputAttrs_;
    std::vector<rknn_input> inputs_;
    std::vector<std::vector<std::uint8_t>> inputBuffers_;
    std::vector<rknn_tensor_attr> outputAttrs_;
    std::vector<rknn_output> outputs_;
    std::vector<cv::Mat> outputCache_;
    std::vector<float> inputScaling_;
    bool outputsAcquired_ = false;
};
