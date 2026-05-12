#pragma once

#include "framework/context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <rknn_api.h>
#include <string>
#include <vector>

class InferContext : public GryFlux::Context
{
public:
    explicit InferContext(
        const std::string &model_path,
        int device_id = 0,
        int expected_model_width = 0,
        int expected_model_height = 0);
    ~InferContext() override;

    int getNpuId() const { return device_id_; }
    int getModelWidth() const { return model_width_; }
    int getModelHeight() const { return model_height_; }
    std::size_t getOutputCount() const { return output_attrs_.size(); }

    void setInput(const uint8_t *data, std::size_t size);
    void runInference();
    std::pair<void *, std::size_t> getOutput(std::size_t output_index);
    const rknn_tensor_attr &getOutputAttr(std::size_t index) const;
    float deqntAffineToF32(int8_t qnt, int zp, float scale) const;

private:
    static rknn_core_mask toCoreMask(int device_id);
    static std::size_t tensorTypeSize(rknn_tensor_type type);
    static std::size_t tensorByteSize(const rknn_tensor_attr &attr);

    void releaseResources();

    int device_id_ = 0;
    rknn_context ctx_ = 0;
    int model_width_ = 0;
    int model_height_ = 0;
    rknn_tensor_type input_type_ = RKNN_TENSOR_UINT8;

    std::vector<std::uint8_t> model_data_;
    rknn_tensor_attr input_attr_{};
    rknn_tensor_attr io_input_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_tensor_attr> output_io_attrs_;
    rknn_tensor_mem *input_mem_ = nullptr;
    std::vector<rknn_tensor_mem *> output_mems_;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateDetectionInferContexts(
    const std::string &model_path,
    int device_id,
    std::size_t instance_count);
