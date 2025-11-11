#pragma once

#include <fstream>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "framework/processing_task.h"
#include "package.h"
#include "rknn_api.h"
#include "utils/logger.h"
#include "opencv2/opencv.hpp"

namespace GryFlux {
namespace RealESRGAN {

using ModelData = std::pair<std::unique_ptr<unsigned char[]>, std::size_t>;

class RkRunner : public GryFlux::ProcessingTask {
public:
    RkRunner(std::string_view model_path,
             int npu_id = 1,
             std::size_t model_width = 256,
             std::size_t model_height = 256);
    ~RkRunner();

    std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

private:
    std::optional<ModelData> load_model(std::string_view filename);
    void dump_tensor_attr(rknn_tensor_attr *attr);
    float deqnt_affine_to_f32(int8_t qnt, int zp, float scale) const;
    cv::Mat makeOutputMat(const std::vector<float> &output, const rknn_tensor_attr &attr) const;

    rknn_context rknn_ctx_;
    std::size_t input_num_;
    std::size_t output_num_;
    rknn_tensor_attr *input_attrs_;
    rknn_tensor_attr *output_attrs_;
    std::vector<rknn_tensor_mem *> input_mems_;
    std::vector<rknn_tensor_mem *> output_mems_;
    std::size_t model_width_;
    std::size_t model_height_;
    bool is_quant_;
};

} // namespace RealESRGAN
} // namespace GryFlux
