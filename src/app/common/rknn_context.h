#pragma once

#include "framework/context.h"

#include <cstddef>
#include <cstdint>
#include <rknn_api.h>
#include <string>
#include <vector>

class RKNNContext : public GryFlux::Context
{
public:
    explicit RKNNContext(int deviceId,
                         const std::string &modelPath,
                         int expectedModelWidth = 0,
                         int expectedModelHeight = 0);
    ~RKNNContext() override;

    int getNpuId() const { return deviceId_; }
    int getModelWidth() const { return modelWidth_; }
    int getModelHeight() const { return modelHeight_; }
    size_t getOutputCount() const { return outputAttrs_.size(); }

    void setInput(const uint8_t *data, size_t size);
    void runInference();
    std::pair<void *, size_t> getOutput(size_t outputIndex);
    const rknn_tensor_attr &getOutputAttr(size_t index) const;
    float deqntAffineToF32(int8_t qnt, int zp, float scale) const;

private:
    static rknn_core_mask toCoreMask(int deviceId);
    static std::size_t tensorTypeSize(rknn_tensor_type type);
    static std::size_t tensorByteSize(const rknn_tensor_attr &attr);

    void releaseResources();

    int deviceId_ = 0;

    rknn_context ctx_ = 0;
    int modelWidth_ = 0;
    int modelHeight_ = 0;
    rknn_tensor_type inputType_ = RKNN_TENSOR_UINT8;

    std::vector<std::uint8_t> modelData_;
    rknn_tensor_attr inputAttr_{};
    rknn_tensor_attr ioInputAttr_{};
    std::vector<rknn_tensor_attr> outputAttrs_;
    std::vector<rknn_tensor_attr> outputIoAttrs_;
    rknn_tensor_mem *inputMem_ = nullptr;
    std::vector<rknn_tensor_mem *> outputMems_;
};
