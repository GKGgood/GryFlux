#include "context/infercontext.h"

#include "utils/logger.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{

constexpr rknn_core_mask kCoreMasks[] = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_0_1,
    RKNN_NPU_CORE_0_1_2,
};

int checkRknnCall(int ret, const char *op)
{
    if (ret < 0)
    {
        throw std::runtime_error(std::string(op) + " failed, ret=" + std::to_string(ret));
    }
    return ret;
}

float dequantizeValue(std::int32_t value, int zero_point, float scale)
{
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

std::uint16_t floatToFp16(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0)
    {
        if (exponent < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> static_cast<std::uint32_t>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }

    if (exponent >= 31)
    {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }

    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10) | ((mantissa + 0x1000u) >> 13));
}

} // namespace

InferContext::InferContext(
    const std::string &model_path,
    int device_id,
    int expected_model_width,
    int expected_model_height)
    : device_id_(device_id)
{
    std::ifstream model_stream(model_path, std::ios::binary);
    if (!model_stream)
    {
        throw std::runtime_error("Failed to open RKNN detection model: " + model_path);
    }

    model_stream.seekg(0, std::ios::end);
    const auto length = model_stream.tellg();
    if (length <= 0)
    {
        throw std::runtime_error("Empty RKNN detection model: " + model_path);
    }

    model_stream.seekg(0, std::ios::beg);
    model_data_.resize(static_cast<std::size_t>(length));
    model_stream.read(
        reinterpret_cast<char *>(model_data_.data()),
        static_cast<std::streamsize>(length));
    if (!model_stream)
    {
        throw std::runtime_error("Failed to read RKNN detection model: " + model_path);
    }

    checkRknnCall(rknn_init(&ctx_, model_data_.data(), model_data_.size(), 0, nullptr), "rknn_init");
    checkRknnCall(rknn_set_core_mask(ctx_, toCoreMask(device_id_)), "rknn_set_core_mask");

    rknn_input_output_num io_num{};
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)), "rknn_query(io_num)");

    if (io_num.n_input != 1)
    {
        throw std::runtime_error("InferContext only supports single-input RKNN models");
    }
    if (io_num.n_output < 1)
    {
        throw std::runtime_error("Detection RKNN model has no outputs");
    }

    std::memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)), "rknn_query(input_attr)");

    if (input_attr_.fmt != RKNN_TENSOR_NHWC)
    {
        throw std::runtime_error("InferContext only supports NHWC RKNN input tensors");
    }

    model_height_ = static_cast<int>(input_attr_.dims[1]);
    model_width_ = static_cast<int>(input_attr_.dims[2]);
    if ((expected_model_width > 0 && model_width_ != expected_model_width) ||
        (expected_model_height > 0 && model_height_ != expected_model_height))
    {
        throw std::runtime_error("Detection RKNN input shape mismatch");
    }

    input_type_ = input_attr_.type;
    if (input_type_ != RKNN_TENSOR_INT8 && input_type_ != RKNN_TENSOR_FLOAT16)
    {
        throw std::runtime_error("Detection RKNN input tensor type must be int8 or fp16");
    }

    io_input_attr_ = input_attr_;
    if (input_type_ == RKNN_TENSOR_INT8)
    {
        io_input_attr_.type = RKNN_TENSOR_UINT8;
        io_input_attr_.fmt = RKNN_TENSOR_NHWC;
    }

    input_mem_ = rknn_create_mem(ctx_, input_attr_.size_with_stride);
    if (!input_mem_)
    {
        throw std::runtime_error("rknn_create_mem(detection_input) failed");
    }
    checkRknnCall(rknn_set_io_mem(ctx_, input_mem_, &io_input_attr_), "rknn_set_io_mem(detection_input)");

    output_attrs_.resize(io_num.n_output);
    output_io_attrs_.resize(io_num.n_output);
    output_mems_.resize(io_num.n_output, nullptr);

    for (std::size_t i = 0; i < output_attrs_.size(); ++i)
    {
        auto &attr = output_attrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = static_cast<std::uint32_t>(i);
        checkRknnCall(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "rknn_query(output_attr)");

        auto &io_attr = output_io_attrs_[i];
        io_attr = attr;

        if (attr.type == RKNN_TENSOR_FLOAT16 || attr.type == RKNN_TENSOR_FLOAT32)
        {
            io_attr.type = RKNN_TENSOR_FLOAT32;
        }
        else if (attr.type != RKNN_TENSOR_INT8)
        {
            throw std::runtime_error("Detection RKNN output tensor type must be int8/fp16/fp32");
        }

        output_mems_[i] = rknn_create_mem(ctx_, tensorByteSize(io_attr));
        if (!output_mems_[i])
        {
            throw std::runtime_error("rknn_create_mem(detection_output) failed");
        }
        checkRknnCall(rknn_set_io_mem(ctx_, output_mems_[i], &io_attr), "rknn_set_io_mem(detection_output)");
    }

    LOG.info(
        "[InferContext] model=%s input=%dx%d fmt=%d outputs=%zu npu=%d",
        model_path.c_str(),
        model_width_,
        model_height_,
        static_cast<int>(input_attr_.fmt),
        output_attrs_.size(),
        device_id_);
}

InferContext::~InferContext()
{
    releaseResources();
}

void InferContext::setInput(const uint8_t *data, std::size_t size)
{
    const std::size_t input_byte_size =
        static_cast<std::size_t>(model_width_) * static_cast<std::size_t>(model_height_) * 3U;
    if (size != input_byte_size)
    {
        throw std::runtime_error("Invalid detection RKNN input tensor byte size");
    }

    if (input_type_ == RKNN_TENSOR_INT8)
    {
        if (io_input_attr_.w_stride == model_width_)
        {
            std::memcpy(input_mem_->virt_addr, data, size);
        }
        else
        {
            const std::size_t src_row_bytes = static_cast<std::size_t>(model_width_) * 3U;
            const std::size_t dst_row_bytes =
                static_cast<std::size_t>(io_input_attr_.w_stride) * 3U;
            auto *src = data;
            auto *dst = reinterpret_cast<std::uint8_t *>(input_mem_->virt_addr);
            for (int h = 0; h < model_height_; ++h)
            {
                std::memcpy(dst, src, src_row_bytes);
                src += src_row_bytes;
                dst += dst_row_bytes;
            }
        }
    }
    else
    {
        auto *dst = reinterpret_cast<std::uint16_t *>(input_mem_->virt_addr);
        const std::size_t dst_row_elems = static_cast<std::size_t>(input_attr_.w_stride) * 3U;
        for (int h = 0; h < model_height_; ++h)
        {
            const std::size_t src_row_base =
                static_cast<std::size_t>(h) * static_cast<std::size_t>(model_width_) * 3U;
            for (int w = 0; w < model_width_; ++w)
            {
                const std::size_t src_index = src_row_base + static_cast<std::size_t>(w) * 3U;
                for (int c = 0; c < 3; ++c)
                {
                    dst[static_cast<std::size_t>(h) * dst_row_elems +
                        static_cast<std::size_t>(w) * 3U +
                        static_cast<std::size_t>(c)] =
                        floatToFp16(static_cast<float>(data[src_index + static_cast<std::size_t>(c)]));
                }
            }
        }
    }

    checkRknnCall(rknn_mem_sync(ctx_, input_mem_, RKNN_MEMORY_SYNC_TO_DEVICE), "rknn_mem_sync(detection_input)");
}

void InferContext::runInference()
{
    checkRknnCall(rknn_run(ctx_, nullptr), "rknn_run(detection)");
    for (std::size_t i = 0; i < output_mems_.size(); ++i)
    {
        checkRknnCall(
            rknn_mem_sync(ctx_, output_mems_[i], RKNN_MEMORY_SYNC_FROM_DEVICE),
            "rknn_mem_sync(detection_output)");
    }
}

std::pair<void *, std::size_t> InferContext::getOutput(std::size_t output_index)
{
    if (output_index >= output_mems_.size())
    {
        throw std::out_of_range("Detection output index out of range");
    }
    return {output_mems_[output_index]->virt_addr, output_attrs_[output_index].n_elems};
}

const rknn_tensor_attr &InferContext::getOutputAttr(std::size_t index) const
{
    if (index >= output_io_attrs_.size())
    {
        throw std::out_of_range("Detection output index out of range");
    }
    return output_io_attrs_[index];
}

float InferContext::deqntAffineToF32(int8_t qnt, int zp, float scale) const
{
    return dequantizeValue(qnt, zp, scale);
}

rknn_core_mask InferContext::toCoreMask(int device_id)
{
    if (device_id < 0)
    {
        device_id = 0;
    }
    if (device_id >= static_cast<int>(sizeof(kCoreMasks) / sizeof(kCoreMasks[0])))
    {
        device_id = static_cast<int>(sizeof(kCoreMasks) / sizeof(kCoreMasks[0])) - 1;
    }
    return kCoreMasks[device_id];
}

std::size_t InferContext::tensorTypeSize(rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_FLOAT32:
        return sizeof(float);
    case RKNN_TENSOR_FLOAT16:
        return sizeof(std::uint16_t);
    case RKNN_TENSOR_INT8:
        return sizeof(std::int8_t);
    default:
        throw std::runtime_error("Unsupported detection RKNN tensor type");
    }
}

std::size_t InferContext::tensorByteSize(const rknn_tensor_attr &attr)
{
    return static_cast<std::size_t>(attr.n_elems) * tensorTypeSize(attr.type);
}

void InferContext::releaseResources()
{
    if (input_mem_)
    {
        rknn_destroy_mem(ctx_, input_mem_);
        input_mem_ = nullptr;
    }

    for (auto *mem : output_mems_)
    {
        if (mem)
        {
            rknn_destroy_mem(ctx_, mem);
        }
    }
    output_mems_.clear();

    if (ctx_ != 0)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateDetectionInferContexts(
    const std::string &model_path,
    int device_id,
    std::size_t instance_count)
{
    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);

    auto probe_context = std::make_shared<InferContext>(model_path, device_id);
    const int model_width = probe_context->getModelWidth();
    const int model_height = probe_context->getModelHeight();
    contexts.push_back(probe_context);

    for (std::size_t i = 1; i < instance_count; ++i)
    {
        contexts.push_back(std::make_shared<InferContext>(
            model_path,
            static_cast<int>(i),
            model_width,
            model_height));
    }

    return contexts;
}
