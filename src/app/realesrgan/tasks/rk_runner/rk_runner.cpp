#include "rk_runner.h"

#include <algorithm>
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
      is_quant_(false) {
    LOG.info("[RealESRGAN::RkRunner] Model path: %s", model_path.data());
    auto model_meta = load_model(model_path);
    if (!model_meta) {
        throw std::runtime_error("Failed to read RKNN model");
    }

    auto &[model_data, model_size] = *model_meta;
    RKNN_CHECK(rknn_init(&rknn_ctx_, model_data.get(), model_size, 0, nullptr), "rknn_init");

    constexpr std::size_t core_count = sizeof(kNpuCores) / sizeof(kNpuCores[0]);
    const int clamped_id = std::clamp(npu_id, 0, static_cast<int>(core_count - 1));
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

    input_attrs_[0].type = RKNN_TENSOR_UINT8;
    input_attrs_[0].fmt = RKNN_TENSOR_NHWC;
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
    LOG.warning("[RealESRGAN::RkRunner] Unexpected input size %dx%d. Expected %zux%zu", input_tensor.cols, input_tensor.rows, model_width_, model_height_);
    }

    std::memcpy(input_mems_[0]->virt_addr, input_tensor.data, input_mems_[0]->size);
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
