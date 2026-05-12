#include "context/fusion_npu_context.h"

#include "utils/logger.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace
{
constexpr rknn_core_mask kCoreMasks[] = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_AUTO,
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

float deqntAffineToF32(std::int32_t value, int zeroPoint, float scale)
{
    return (static_cast<float>(value) - static_cast<float>(zeroPoint)) * scale;
}
} // namespace

FusionNpuContext::FusionNpuContext(int deviceId,
                                   const std::string &modelPath,
                                   int expectedModelWidth,
                                   int expectedModelHeight)
    : deviceId_(deviceId),
      expectedModelWidth_(expectedModelWidth),
      expectedModelHeight_(expectedModelHeight)
{
    loadModel(modelPath);

    checkRknnCall(rknn_init(&ctx_, modelData_.data(), modelData_.size(), 0, nullptr), "rknn_init");
    checkRknnCall(rknn_set_core_mask(ctx_, toCoreMask(deviceId_)), "rknn_set_core_mask");

    prepareInputTensors();
    prepareOutputTensors();
}

FusionNpuContext::~FusionNpuContext()
{
    releaseResources();
}

void FusionNpuContext::setInputs(const cv::Mat &visibleYF32, const cv::Mat &infraredF32)
{
    validateInput(visibleYF32, "visible Y");
    validateInput(infraredF32, "infrared");

    releaseOutputs();
    setInput(0, visibleYF32);
    setInput(1, infraredF32);
    checkRknnCall(rknn_inputs_set(ctx_,
                                  static_cast<std::uint32_t>(inputs_.size()),
                                  inputs_.data()),
                  "rknn_inputs_set");
}

void FusionNpuContext::runInference()
{
    checkRknnCall(rknn_run(ctx_, nullptr), "rknn_run");
    checkRknnCall(rknn_outputs_get(ctx_,
                                   static_cast<std::uint32_t>(outputs_.size()),
                                   outputs_.data(),
                                   nullptr),
                  "rknn_outputs_get");
    outputsAcquired_ = true;

    for (std::size_t i = 0; i < outputs_.size(); ++i)
    {
        const auto &attr = outputAttrs_[i];
        int height = 0;
        int width = 0;
        resolveSpatial(attr, height, width);

        cv::Mat output(height, width, CV_32FC1);
        std::memcpy(output.data,
                    outputs_[i].buf,
                    static_cast<std::size_t>(attr.n_elems) * sizeof(float));
        outputCache_[i] = std::move(output);
    }
}

cv::Mat FusionNpuContext::getOutput(std::size_t index)
{
    if (index >= outputCache_.size())
    {
        throw std::out_of_range("FusionNetV2 output index out of range");
    }
    if (outputCache_[index].empty())
    {
        throw std::runtime_error("FusionNetV2 output is empty");
    }
    return outputCache_[index].clone();
}

rknn_core_mask FusionNpuContext::toCoreMask(int deviceId)
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

float FusionNpuContext::inputScaleForAttr(const rknn_tensor_attr &attr)
{
    if (attr.type == RKNN_TENSOR_FLOAT16 &&
        attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        std::fabs(attr.scale - 1.0f) < 1e-4f &&
        attr.zp == 0)
    {
        return 255.0f;
    }

    return 1.0f;
}

std::size_t FusionNpuContext::tensorTypeSize(rknn_tensor_type type)
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

void FusionNpuContext::resolveSpatial(const rknn_tensor_attr &attr, int &height, int &width)
{
    if (attr.n_dims != 4)
    {
        throw std::runtime_error("FusionNetV2 expects 4D RKNN tensors");
    }

    if (attr.fmt == RKNN_TENSOR_NCHW)
    {
        height = static_cast<int>(attr.dims[2]);
        width = static_cast<int>(attr.dims[3]);
        return;
    }

    height = static_cast<int>(attr.dims[1]);
    width = static_cast<int>(attr.dims[2]);
}

void FusionNpuContext::dumpTensorAttr(const rknn_tensor_attr &attr)
{
    LOG.info("[FusionNpuContext] index=%d name=%s dims=[%d,%d,%d,%d] n_elems=%u type=%s qnt_type=%s zp=%d scale=%f",
             attr.index,
             attr.name,
             attr.dims[0],
             attr.dims[1],
             attr.dims[2],
             attr.dims[3],
             static_cast<unsigned int>(attr.n_elems),
             get_type_string(attr.type),
             get_qnt_type_string(attr.qnt_type),
             attr.zp,
             attr.scale);
}

void FusionNpuContext::loadModel(const std::string &path)
{
    std::ifstream modelStream(path, std::ios::binary);
    if (!modelStream)
    {
        throw std::runtime_error("Failed to open RKNN model: " + path);
    }

    modelStream.seekg(0, std::ios::end);
    const auto length = modelStream.tellg();
    if (length <= 0)
    {
        throw std::runtime_error("Empty RKNN model: " + path);
    }

    modelStream.seekg(0, std::ios::beg);
    modelData_.resize(static_cast<std::size_t>(length));
    modelStream.read(reinterpret_cast<char *>(modelData_.data()), static_cast<std::streamsize>(length));
    if (!modelStream)
    {
        throw std::runtime_error("Failed to read RKNN model: " + path);
    }
}

void FusionNpuContext::prepareInputTensors()
{
    rknn_input_output_num ioNum{};
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "rknn_query(io_num)");

    if (ioNum.n_input != 2)
    {
        throw std::runtime_error("FusionNetV2 only supports dual-input RKNN models");
    }
    if (ioNum.n_output < 1)
    {
        throw std::runtime_error("FusionNetV2 model has no outputs");
    }

    inputAttrs_.resize(ioNum.n_input);
    inputs_.resize(ioNum.n_input);
    inputBuffers_.resize(ioNum.n_input);
    inputScaling_.resize(ioNum.n_input, 1.0f);

    for (std::size_t i = 0; i < inputAttrs_.size(); ++i)
    {
        auto &attr = inputAttrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = static_cast<std::uint32_t>(i);
        checkRknnCall(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "rknn_query(input_attr)");
        dumpTensorAttr(attr);

        if (attr.type != RKNN_TENSOR_INT8 && attr.type != RKNN_TENSOR_FLOAT16)
        {
            throw std::runtime_error("FusionNetV2 input tensor type must be int8 or fp16");
        }

        int height = 0;
        int width = 0;
        resolveSpatial(attr, height, width);
        if (i == 0)
        {
            modelHeight_ = height;
            modelWidth_ = width;
        }
        else if (height != modelHeight_ || width != modelWidth_)
        {
            throw std::runtime_error("FusionNetV2 inputs must share the same spatial shape");
        }

        if ((expectedModelWidth_ > 0 && width != expectedModelWidth_) ||
            (expectedModelHeight_ > 0 && height != expectedModelHeight_))
        {
            throw std::runtime_error("FusionNetV2 input shape mismatch");
        }

        inputScaling_[i] = inputScaleForAttr(attr);
        auto &input = inputs_[i];
        std::memset(&input, 0, sizeof(input));
        input.index = static_cast<std::uint32_t>(i);
        input.type = RKNN_TENSOR_FLOAT32;
        input.fmt = attr.fmt;
        input.pass_through = 0;
    }
}

void FusionNpuContext::prepareOutputTensors()
{
    rknn_input_output_num ioNum{};
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "rknn_query(io_num)");

    outputAttrs_.resize(ioNum.n_output);
    outputs_.resize(ioNum.n_output);
    outputCache_.resize(ioNum.n_output);

    for (std::size_t i = 0; i < outputAttrs_.size(); ++i)
    {
        auto &attr = outputAttrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = static_cast<std::uint32_t>(i);
        checkRknnCall(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "rknn_query(output_attr)");
        dumpTensorAttr(attr);

        auto &output = outputs_[i];
        std::memset(&output, 0, sizeof(output));
        output.want_float = 1;
        output.is_prealloc = 0;
    }
}

void FusionNpuContext::releaseOutputs()
{
    if (!outputsAcquired_)
    {
        return;
    }

    rknn_outputs_release(ctx_,
                         static_cast<std::uint32_t>(outputs_.size()),
                         outputs_.data());
    outputsAcquired_ = false;

    for (auto &output : outputCache_)
    {
        output.release();
    }
}

void FusionNpuContext::releaseResources()
{
    releaseOutputs();
    if (ctx_ != 0)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

void FusionNpuContext::validateInput(const cv::Mat &mat, const char *name) const
{
    if (mat.empty())
    {
        throw std::runtime_error(std::string("FusionNetV2 received empty ") + name + " input");
    }
    if (mat.type() != CV_32FC1)
    {
        throw std::runtime_error(std::string("FusionNetV2 expects ") + name + " as CV_32FC1");
    }
    if (mat.cols != modelWidth_ || mat.rows != modelHeight_)
    {
        throw std::runtime_error(std::string("FusionNetV2 ") + name + " input size mismatch");
    }
}

void FusionNpuContext::setInput(std::size_t index, const cv::Mat &mat)
{
    if (index >= inputAttrs_.size() || index >= inputs_.size() || index >= inputBuffers_.size())
    {
        throw std::out_of_range("FusionNetV2 input index out of range");
    }

    const auto &attr = inputAttrs_[index];
    auto &input = inputs_[index];
    auto &buffer = inputBuffers_[index];

    cv::Mat scaledMat = mat;
    if (!scaledMat.isContinuous())
    {
        scaledMat = scaledMat.clone();
    }

    const float inputScale = inputScaling_[index];
    if (std::fabs(inputScale - 1.0f) > 1e-6f)
    {
        scaledMat *= inputScale;
    }

    if (static_cast<std::size_t>(scaledMat.total()) != attr.n_elems)
    {
        throw std::runtime_error("FusionNetV2 input element count mismatch");
    }

    buffer.resize(static_cast<std::size_t>(attr.n_elems) * sizeof(float));
    std::memcpy(buffer.data(), scaledMat.ptr<float>(), buffer.size());
    input.buf = buffer.data();
    input.size = static_cast<std::uint32_t>(buffer.size());
}
