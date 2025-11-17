#include "rk_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace GryFlux {
namespace RealESRGAN {

#define RKNN_CHECK(op, error_msg)                    \
    do {                                             \
        int ret = (op);                              \
        if (ret < 0) {                               \
            LOG.error("%s failed! ret=%d", error_msg, ret); \
            throw std::runtime_error(error_msg);      \
        }                                             \
    } while (0)

namespace {
constexpr rknn_core_mask kNpuCores[] = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_0_1,
    RKNN_NPU_CORE_0_1_2
};
}

RkRunner::RkRunner(std::string_view model_path,
                   int npu_id,
                   std::size_t model_width,
                   std::size_t model_height)
    : input_num_(0),
      output_num_(0),
      input_attrs_(nullptr),
      output_attrs_(nullptr),
      model_width_(model_width),
      model_height_(model_height),
    is_quant_(false),
    input_type_(RKNN_TENSOR_FLOAT32),
    input_quantized_(false),
    input_channels_(0),
    input_element_size_(0),
    input_scale_(1.0f),
    input_zero_point_(0) {
    LOG.info("[RealESRGAN::RkRunner] Model path: %s", model_path.data());
    auto model_meta = load_model(model_path);
    if (!model_meta) {
        throw std::runtime_error("Failed to read RKNN model");
    }

    auto &[model_data, model_size] = *model_meta;
    RKNN_CHECK(rknn_init(&rknn_ctx_, model_data.get(), model_size, 0, nullptr), "rknn_init");

    constexpr std::size_t core_count = sizeof(kNpuCores) / sizeof(kNpuCores[0]);
    const int clamped_id = std::max(0, std::min(npu_id, static_cast<int>(core_count - 1)));
    RKNN_CHECK(rknn_set_core_mask(rknn_ctx_, kNpuCores[static_cast<std::size_t>(clamped_id)]), "set NPU core mask");

    rknn_sdk_version version{};
    RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)), "query rknn version");
    LOG.info("[RealESRGAN::RkRunner] rknn sdk version: %s, driver version: %s", version.api_version, version.drv_version);

    rknn_input_output_num io_num{};
    RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)), "query io num");
    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;

    input_attrs_ = new rknn_tensor_attr[input_num_];
    std::memset(input_attrs_, 0, sizeof(rknn_tensor_attr) * input_num_);
    for (std::size_t i = 0; i < input_num_; ++i) {
        input_attrs_[i].index = static_cast<uint32_t>(i);
        RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr)),
                   "query input attr");
        dump_tensor_attr(&input_attrs_[i]);
    }

    output_attrs_ = new rknn_tensor_attr[output_num_];
    std::memset(output_attrs_, 0, sizeof(rknn_tensor_attr) * output_num_);
    for (std::size_t i = 0; i < output_num_; ++i) {
        output_attrs_[i].index = static_cast<uint32_t>(i);
        RKNN_CHECK(rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr)),
                   "query output attr");
        dump_tensor_attr(&output_attrs_[i]);
    }

    is_quant_ = (output_num_ > 0 && output_attrs_[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                 output_attrs_[0].type != RKNN_TENSOR_FLOAT16);
    if (is_quant_) {
        LOG.info("[RealESRGAN::RkRunner] Quantized model detected");
    } else {
        LOG.info("[RealESRGAN::RkRunner] Floating-point model detected");
    }

    if (input_num_ == 0) {
        throw std::runtime_error("Model has no input tensors");
    }

    if (input_attrs_[0].fmt != RKNN_TENSOR_NHWC) {
        throw std::runtime_error("Only NHWC input tensor is supported in zero-copy mode");
    }

    input_type_ = input_attrs_[0].type;
    input_quantized_ = ((input_type_ == RKNN_TENSOR_UINT8 || input_type_ == RKNN_TENSOR_INT8) &&
                        input_attrs_[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);

    input_scale_ = input_attrs_[0].scale;
    input_zero_point_ = input_attrs_[0].zp;

    if (input_quantized_) {
        LOG.info("[RealESRGAN::RkRunner] Quantized input tensor detected (type=%s, zp=%d, scale=%f)",
                 get_type_string(input_type_),
                 input_zero_point_,
                 input_scale_);
    } else {
        LOG.info("[RealESRGAN::RkRunner] Floating-point input tensor detected (type=%s)",
                 get_type_string(input_type_));
    }

    if (input_attrs_[0].n_dims >= 4) {
        input_channels_ = static_cast<std::size_t>(input_attrs_[0].dims[3]);
    } else if (input_attrs_[0].n_dims == 3) {
        input_channels_ = static_cast<std::size_t>(input_attrs_[0].dims[2]);
    } else {
        input_channels_ = 3;
    }

    switch (input_type_) {
    case RKNN_TENSOR_UINT8:
        input_element_size_ = sizeof(uint8_t);
        break;
    case RKNN_TENSOR_INT8:
        input_element_size_ = sizeof(int8_t);
        break;
    case RKNN_TENSOR_FLOAT16:
        input_element_size_ = sizeof(uint16_t);
        break;
    case RKNN_TENSOR_FLOAT32:
        input_element_size_ = sizeof(float);
        break;
    default:
        LOG.error("[RealESRGAN::RkRunner] Unsupported input tensor type: %d", static_cast<int>(input_type_));
        throw std::runtime_error("Unsupported input tensor type");
    }

    input_attrs_[0].fmt = RKNN_TENSOR_NHWC;
    input_attrs_[0].type = input_type_;
    input_mems_.emplace_back(rknn_create_mem(rknn_ctx_, input_attrs_[0].size_with_stride));
    RKNN_CHECK(rknn_set_io_mem(rknn_ctx_, input_mems_[0], &input_attrs_[0]), "set input mem");

    for (std::size_t i = 0; i < output_num_; ++i) {
        std::size_t output_size = 0;
        if (is_quant_) {
            output_attrs_[i].type = RKNN_TENSOR_INT8;
            output_size = output_attrs_[i].n_elems * sizeof(int8_t);
        } else {
            output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
            output_size = output_attrs_[i].n_elems * sizeof(float);
        }

        output_mems_.emplace_back(rknn_create_mem(rknn_ctx_, output_size));
        RKNN_CHECK(rknn_set_io_mem(rknn_ctx_, output_mems_[i], &output_attrs_[i]), "set output mem");
    }
}

RkRunner::~RkRunner() {
    for (auto *mem : input_mems_) {
        rknn_destroy_mem(rknn_ctx_, mem);
    }
    for (auto *mem : output_mems_) {
        rknn_destroy_mem(rknn_ctx_, mem);
    }
    delete[] input_attrs_;
    delete[] output_attrs_;
    rknn_destroy(rknn_ctx_);
}

std::optional<ModelData> RkRunner::load_model(std::string_view filename) {
    std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG.error("[RealESRGAN::RkRunner] Failed to open model file: %s", filename.data());
        return std::nullopt;
    }

    const std::size_t file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    auto buffer = std::make_unique<unsigned char[]>(file_size);
    file.read(reinterpret_cast<char *>(buffer.get()), static_cast<std::streamsize>(file_size));
    file.close();
    return ModelData{std::move(buffer), file_size};
}

void RkRunner::dump_tensor_attr(rknn_tensor_attr *attr) {
    LOG.info("[RealESRGAN::RkRunner] index=%d, name=%s, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt=%s, zp=%d, scale=%f",
             attr->index,
             attr->name,
             attr->dims[0],
             attr->dims[1],
             attr->dims[2],
             attr->dims[3],
             attr->n_elems,
             attr->size,
             get_format_string(attr->fmt),
             get_type_string(attr->type),
             get_qnt_type_string(attr->qnt_type),
             attr->zp,
             attr->scale);
}

float RkRunner::deqnt_affine_to_f32(int8_t qnt, int zp, float scale) const {
    return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale;
}

cv::Mat RkRunner::makeOutputMat(const std::vector<float> &output, const rknn_tensor_attr &attr) const {
    int batch = (attr.n_dims > 0) ? static_cast<int>(attr.dims[0]) : 1;
    if (batch != 1) {
        LOG.warning("[RealESRGAN::RkRunner] Only batch size 1 is supported, got %d", batch);
    }

    int channels = 3;
    int height = static_cast<int>(model_height_);
    int width = static_cast<int>(model_width_);

    if (attr.fmt == RKNN_TENSOR_NHWC) {
        if (attr.n_dims >= 4) {
            height = static_cast<int>(attr.dims[1]);
            width = static_cast<int>(attr.dims[2]);
            channels = static_cast<int>(attr.dims[3]);
        }
        const int type = CV_32FC(channels);
        cv::Mat nhwc(height, width, type, const_cast<float *>(output.data()));
        return nhwc.clone();
    }

    if (attr.n_dims >= 4) {
        channels = static_cast<int>(attr.dims[1]);
        height = static_cast<int>(attr.dims[2]);
        width = static_cast<int>(attr.dims[3]);
    }

    const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::vector<cv::Mat> channel_mats;
    channel_mats.reserve(channels);
    for (int c = 0; c < channels; ++c) {
        const float *ptr = output.data() + static_cast<std::size_t>(c) * plane;
        cv::Mat channel(height, width, CV_32F, const_cast<float *>(ptr));
        channel_mats.emplace_back(channel.clone());
    }

    cv::Mat merged;
    cv::merge(channel_mats, merged);
    return merged;
}

std::shared_ptr<DataObject> RkRunner::process(const std::vector<std::shared_ptr<DataObject>> &inputs) {
    if (inputs.size() != 1) {
        LOG.error("[RealESRGAN::RkRunner] Invalid input size: %zu", inputs.size());
        return nullptr;
    }

    auto input_pkg = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
    if (!input_pkg) {
        LOG.error("[RealESRGAN::RkRunner] Input cast failed");
        return nullptr;
    }

    const cv::Mat &input_tensor = input_pkg->get_data();
    if (input_tensor.cols != static_cast<int>(model_width_) || input_tensor.rows != static_cast<int>(model_height_)) {
        LOG.warning("[RealESRGAN::RkRunner] Unexpected input size %dx%d. Expected %zux%zu",
                    input_tensor.cols,
                    input_tensor.rows,
                    model_width_,
                    model_height_);
    }

    cv::Mat prepared;
    const int channels = static_cast<int>(input_channels_ > 0 ? input_channels_ : input_tensor.channels());
    if (input_quantized_) {
        const int float_type = CV_32F;
        cv::Mat normalized;
        input_tensor.convertTo(normalized, float_type, 1.0f / 255.0f);
        cv::Mat clipped_low;
        cv::max(normalized, 0.0f, clipped_low);
        cv::Mat clipped;
        cv::min(clipped_low, 1.0f, clipped);

        const int mat_type = (input_type_ == RKNN_TENSOR_UINT8) ? CV_MAKETYPE(CV_8U, channels)
                                                                : CV_MAKETYPE(CV_8S, channels);
        prepared.create(clipped.rows, clipped.cols, mat_type);

        const float inv_scale = (input_scale_ == 0.0f) ? 0.0f : 1.0f / input_scale_;
        const int min_val = (input_type_ == RKNN_TENSOR_UINT8) ? 0 : -128;
        const int max_val = (input_type_ == RKNN_TENSOR_UINT8) ? 255 : 127;

        for (int r = 0; r < clipped.rows; ++r) {
            const float *src_row = clipped.ptr<float>(r);
            if (input_type_ == RKNN_TENSOR_UINT8) {
                auto *dst_row = prepared.ptr<uint8_t>(r);
                for (int c = 0; c < clipped.cols; ++c) {
                    for (int ch = 0; ch < channels; ++ch) {
                        const float real_val = src_row[c * channels + ch];
                        const float quant = std::round(real_val * inv_scale + static_cast<float>(input_zero_point_));
                        const int raw_val = static_cast<int>(quant);
                        const int clamped = std::max(min_val, std::min(max_val, raw_val));
                        dst_row[c * channels + ch] = static_cast<uint8_t>(clamped);
                    }
                }
            } else {
                auto *dst_row = prepared.ptr<int8_t>(r);
                for (int c = 0; c < clipped.cols; ++c) {
                    for (int ch = 0; ch < channels; ++ch) {
                        const float real_val = src_row[c * channels + ch];
                        const float quant = std::round(real_val * inv_scale + static_cast<float>(input_zero_point_));
                        const int raw_val = static_cast<int>(quant);
                        const int clamped = std::max(min_val, std::min(max_val, raw_val));
                        dst_row[c * channels + ch] = static_cast<int8_t>(clamped);
                    }
                }
            }
        }
    } else {
        const int float_type = CV_32F;
        const bool expect_unit_range = (input_type_ == RKNN_TENSOR_FLOAT32);
        const float alpha = expect_unit_range ? (1.0f / 255.0f) : 1.0f;

        cv::Mat scaled;
        input_tensor.convertTo(scaled, float_type, alpha);

        const float upper = expect_unit_range ? 1.0f : 255.0f;
        cv::max(scaled, 0.0f, scaled);
        cv::min(scaled, upper, scaled);

        if (input_type_ == RKNN_TENSOR_FLOAT16) {
            scaled.convertTo(prepared, CV_MAKETYPE(CV_16F, channels));
        } else {
            prepared = scaled;
        }
    }

    if (!prepared.isContinuous()) {
        prepared = prepared.clone();
    }

    if (prepared.empty()) {
        LOG.error("[RealESRGAN::RkRunner] Prepared input tensor is empty after conversion");
        return nullptr;
    }

    if (static_cast<std::size_t>(prepared.channels()) != input_channels_) {
        LOG.warning("[RealESRGAN::RkRunner] Channel mismatch: prepared=%d expected=%zu",
                    prepared.channels(),
                    input_channels_);
    }

    const std::size_t expected_bytes = static_cast<std::size_t>(input_attrs_[0].n_elems) * input_element_size_;

    const std::size_t actual_bytes = static_cast<std::size_t>(prepared.total()) * prepared.elemSize();
    if (actual_bytes < expected_bytes) {
        LOG.error("[RealESRGAN::RkRunner] Prepared input bytes (%zu) less than expected (%zu)", actual_bytes, expected_bytes);
        return nullptr;
    }

    if (input_mems_[0]->size < expected_bytes) {
        LOG.error("[RealESRGAN::RkRunner] Allocated input buffer (%zu) smaller than expected (%zu)",
                  input_mems_[0]->size,
                  expected_bytes);
        return nullptr;
    }

    std::memcpy(input_mems_[0]->virt_addr, prepared.data, expected_bytes);
    RKNN_CHECK(rknn_mem_sync(rknn_ctx_, input_mems_[0], RKNN_MEMORY_SYNC_TO_DEVICE), "sync input");

    RKNN_CHECK(rknn_run(rknn_ctx_, nullptr), "rknn run");

    rknn_tensor_attr &attr = output_attrs_[0];
    RKNN_CHECK(rknn_mem_sync(rknn_ctx_, output_mems_[0], RKNN_MEMORY_SYNC_FROM_DEVICE), "sync output");

    std::vector<float> output(attr.n_elems);
    if (is_quant_) {
        auto *src = reinterpret_cast<int8_t *>(output_mems_[0]->virt_addr);
            for (uint32_t i = 0; i < attr.n_elems; ++i) {
            output[static_cast<std::size_t>(i)] = deqnt_affine_to_f32(src[i], attr.zp, attr.scale);
        }
    } else {
        auto *src = reinterpret_cast<float *>(output_mems_[0]->virt_addr);
        std::memcpy(output.data(), src, static_cast<std::size_t>(attr.n_elems) * sizeof(float));
    }

    cv::Mat sr_tensor = makeOutputMat(output, attr);
    return std::make_shared<SuperResolutionPackage>(sr_tensor);
}

#undef RKNN_CHECK

} // namespace RealESRGAN
} // namespace GryFlux
