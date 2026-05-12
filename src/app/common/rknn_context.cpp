#include "app/common/rknn_context.h"

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

    float dequantizeValue(std::int32_t value, int zeroPoint, float scale)
    {
        return (static_cast<float>(value) - static_cast<float>(zeroPoint)) * scale;
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

        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | ((mantissa + 0x1000u) >> 13));
    }
} // namespace

RKNNContext::RKNNContext(int deviceId,
                         const std::string &modelPath,
                         int expectedModelWidth,
                         int expectedModelHeight)
    : deviceId_(deviceId)
{
    std::ifstream modelStream(modelPath, std::ios::binary);
    if (!modelStream)
    {
        throw std::runtime_error("Failed to open RKNN model: " + modelPath);
    }

    modelStream.seekg(0, std::ios::end);
    const auto length = modelStream.tellg();
    if (length <= 0)
    {
        throw std::runtime_error("Empty RKNN model: " + modelPath);
    }

    modelStream.seekg(0, std::ios::beg);
    modelData_.resize(static_cast<std::size_t>(length));
    modelStream.read(reinterpret_cast<char *>(modelData_.data()), static_cast<std::streamsize>(length));
    if (!modelStream)
    {
        throw std::runtime_error("Failed to read RKNN model: " + modelPath);
    }

    checkRknnCall(rknn_init(&ctx_, modelData_.data(), modelData_.size(), 0, nullptr), "rknn_init");
    checkRknnCall(rknn_set_core_mask(ctx_, toCoreMask(deviceId_)), "rknn_set_core_mask");

    rknn_input_output_num ioNum{};
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "rknn_query(io_num)");

    if (ioNum.n_input != 1)
    {
        throw std::runtime_error("RKNNContext only supports single-input RKNN models");
    }
    if (ioNum.n_output < 1)
    {
        throw std::runtime_error("RKNN model has no outputs");
    }

    std::memset(&inputAttr_, 0, sizeof(inputAttr_));
    inputAttr_.index = 0;
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inputAttr_, sizeof(inputAttr_)), "rknn_query(input_attr)");

    if (inputAttr_.fmt != RKNN_TENSOR_NHWC)
    {
        throw std::runtime_error("RKNNContext only supports RKNN input tensor format NHWC in zero-copy mode");
    }

    modelHeight_ = static_cast<int>(inputAttr_.dims[1]);
    modelWidth_ = static_cast<int>(inputAttr_.dims[2]);

    if ((expectedModelWidth > 0 && modelWidth_ != expectedModelWidth) ||
        (expectedModelHeight > 0 && modelHeight_ != expectedModelHeight))
    {
        throw std::runtime_error("RKNN input shape mismatch");
    }

    inputType_ = inputAttr_.type;
    if (inputType_ != RKNN_TENSOR_INT8 &&
        inputType_ != RKNN_TENSOR_FLOAT16)
    {
        throw std::runtime_error("RKNN input tensor type must be int8 or fp16");
    }

    ioInputAttr_ = inputAttr_;
    if (inputType_ == RKNN_TENSOR_INT8)
    {
        ioInputAttr_.type = RKNN_TENSOR_UINT8;
        ioInputAttr_.fmt = RKNN_TENSOR_NHWC;
    }

    inputMem_ = rknn_create_mem(ctx_, inputAttr_.size_with_stride);
    if (!inputMem_)
    {
        throw std::runtime_error("rknn_create_mem(input) failed");
    }
    checkRknnCall(rknn_set_io_mem(ctx_, inputMem_, &ioInputAttr_), "rknn_set_io_mem(input)");

    LOG.info("RKNN input: fmt=%d dims=[%d,%d,%d,%d] n_dims=%u type=%d qnt_type=%d zp=%d scale=%f size=%u size_with_stride=%u expected_bytes=%zu",
             static_cast<int>(inputAttr_.fmt),
             static_cast<int>(inputAttr_.dims[0]),
             static_cast<int>(inputAttr_.dims[1]),
             static_cast<int>(inputAttr_.dims[2]),
             static_cast<int>(inputAttr_.dims[3]),
             static_cast<unsigned int>(inputAttr_.n_dims),
             static_cast<int>(inputAttr_.type),
             static_cast<int>(inputAttr_.qnt_type),
             inputAttr_.zp,
             static_cast<double>((inputAttr_.scale == 0.0f) ? 1.0f : inputAttr_.scale),
             static_cast<unsigned int>(inputAttr_.size),
             static_cast<unsigned int>(inputAttr_.size_with_stride),
             tensorByteSize(inputAttr_));

    outputAttrs_.resize(ioNum.n_output);
    outputIoAttrs_.resize(ioNum.n_output);
    outputMems_.resize(ioNum.n_output, nullptr);
    for (std::size_t i = 0; i < outputAttrs_.size(); ++i)
    {
        auto &attr = outputAttrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = static_cast<std::uint32_t>(i);
        checkRknnCall(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "rknn_query(output_attr)");

        auto &ioAttr = outputIoAttrs_[i];
        ioAttr = attr;

        if (attr.type == RKNN_TENSOR_INT8)
        {
        }
        else if (attr.type == RKNN_TENSOR_FLOAT16 || attr.type == RKNN_TENSOR_FLOAT32)
        {
            ioAttr.type = RKNN_TENSOR_FLOAT32;
        }
        else
        {
            throw std::runtime_error("RKNN output tensor type must be int8, fp16 or fp32");
        }

        outputMems_[i] = rknn_create_mem(ctx_, tensorByteSize(ioAttr));
        if (!outputMems_[i])
        {
            throw std::runtime_error("rknn_create_mem(output) failed");
        }
        checkRknnCall(rknn_set_io_mem(ctx_, outputMems_[i], &ioAttr), "rknn_set_io_mem(output)");

        LOG.info("RKNN output[%zu]: fmt=%d dims=[%d,%d,%d,%d] n_dims=%u n_elems=%u type=%d io_type=%d qnt_type=%d zp=%d scale=%f size=%u size_with_stride=%u expected_bytes=%zu io_expected_bytes=%zu",
                 i,
                 static_cast<int>(attr.fmt),
                 static_cast<int>(attr.dims[0]),
                 static_cast<int>(attr.dims[1]),
                 static_cast<int>(attr.dims[2]),
                 static_cast<int>(attr.dims[3]),
                 static_cast<unsigned int>(attr.n_dims),
                 static_cast<unsigned int>(attr.n_elems),
                 static_cast<int>(attr.type),
                 static_cast<int>(ioAttr.type),
                 static_cast<int>(attr.qnt_type),
                 attr.zp,
                 static_cast<double>(attr.scale),
                 static_cast<unsigned int>(attr.size),
                 static_cast<unsigned int>(attr.size_with_stride),
                 tensorByteSize(attr),
                 tensorByteSize(ioAttr));
    }

}

RKNNContext::~RKNNContext()
{
    releaseResources();
}

void RKNNContext::setInput(const uint8_t *data, size_t size)
{
    const std::size_t inputByteSize = static_cast<std::size_t>(modelWidth_) * static_cast<std::size_t>(modelHeight_) * 3;
    if (size != inputByteSize)
    {
        throw std::runtime_error("Invalid RKNN input tensor byte size");
    }

    const bool int8Input = (inputType_ == RKNN_TENSOR_INT8);
    const bool fp16Input = (inputType_ == RKNN_TENSOR_FLOAT16);
    if (!int8Input && !fp16Input)
    {
        throw std::runtime_error("Unsupported RKNN input tensor type");
    }

    if (int8Input)
    {
        if (ioInputAttr_.w_stride == modelWidth_)
        {
            std::memcpy(inputMem_->virt_addr, data, size);
        }
        else
        {
            const std::size_t srcRowBytes = static_cast<std::size_t>(modelWidth_) * 3;
            const std::size_t dstRowBytes = static_cast<std::size_t>(ioInputAttr_.w_stride) * 3;
            auto *src = data;
            auto *dst = reinterpret_cast<std::uint8_t *>(inputMem_->virt_addr);
            for (int h = 0; h < modelHeight_; ++h)
            {
                std::memcpy(dst, src, srcRowBytes);
                src += srcRowBytes;
                dst += dstRowBytes;
            }
        }
    }
    else
    {
        auto *dst = reinterpret_cast<std::uint16_t *>(inputMem_->virt_addr);
        const std::size_t dstRowElems = static_cast<std::size_t>(inputAttr_.w_stride) * 3;
        for (int h = 0; h < modelHeight_; ++h)
        {
            const std::size_t srcRowBase = static_cast<std::size_t>(h) * static_cast<std::size_t>(modelWidth_) * 3;
            for (int w = 0; w < modelWidth_; ++w)
            {
                const std::size_t srcIndex = srcRowBase + static_cast<std::size_t>(w) * 3;
                for (int c = 0; c < 3; ++c)
                {
                    dst[static_cast<std::size_t>(h) * dstRowElems + static_cast<std::size_t>(w) * 3 + static_cast<std::size_t>(c)] =
                        floatToFp16(static_cast<float>(data[srcIndex + static_cast<std::size_t>(c)]));
                }
            }
        }
    }
    checkRknnCall(rknn_mem_sync(ctx_, inputMem_, RKNN_MEMORY_SYNC_TO_DEVICE), "rknn_mem_sync(input)");
}

void RKNNContext::runInference()
{
    checkRknnCall(rknn_run(ctx_, nullptr), "rknn_run");
    for (std::size_t i = 0; i < outputMems_.size(); ++i)
    {
        checkRknnCall(rknn_mem_sync(ctx_, outputMems_[i], RKNN_MEMORY_SYNC_FROM_DEVICE), "rknn_mem_sync(output)");
    }
}

std::pair<void *, size_t> RKNNContext::getOutput(size_t outputIndex)
{
    if (outputIndex >= outputMems_.size())
    {
        throw std::out_of_range("Output index out of range");
    }
    return {outputMems_[outputIndex]->virt_addr, outputAttrs_[outputIndex].n_elems};
}

const rknn_tensor_attr &RKNNContext::getOutputAttr(size_t index) const
{
    if (index >= outputIoAttrs_.size())
    {
        throw std::out_of_range("Output index out of range");
    }
    return outputIoAttrs_[index];
}

float RKNNContext::deqntAffineToF32(int8_t qnt, int zp, float scale) const
{
    return dequantizeValue(qnt, zp, scale);
}

rknn_core_mask RKNNContext::toCoreMask(int deviceId)
{
    if (deviceId < 0)
    {
        deviceId = 0;
    }
    if (deviceId >= static_cast<int>(sizeof(kCoreMasks) / sizeof(kCoreMasks[0])))
    {
        deviceId = static_cast<int>(sizeof(kCoreMasks) / sizeof(kCoreMasks[0])) - 1;
    }
    return kCoreMasks[deviceId];
}

std::size_t RKNNContext::tensorTypeSize(rknn_tensor_type type)
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
        throw std::runtime_error("Unsupported RKNN tensor type");
    }
}

std::size_t RKNNContext::tensorByteSize(const rknn_tensor_attr &attr)
{
    return static_cast<std::size_t>(attr.n_elems) * tensorTypeSize(attr.type);
}

void RKNNContext::releaseResources()
{
    if (inputMem_)
    {
        rknn_destroy_mem(ctx_, inputMem_);
        inputMem_ = nullptr;
    }
    for (auto *mem : outputMems_)
    {
        if (mem)
        {
            rknn_destroy_mem(ctx_, mem);
        }
    }
    outputMems_.clear();

    if (ctx_ != 0)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}
