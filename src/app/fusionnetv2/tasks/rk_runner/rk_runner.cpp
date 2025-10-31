#include "rk_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    namespace
    {
        constexpr rknn_core_mask kCoreMask[] = {
            RKNN_NPU_CORE_0,
            RKNN_NPU_CORE_1,
            RKNN_NPU_CORE_2,
            RKNN_NPU_CORE_0_1,
            RKNN_NPU_CORE_0_1_2};

        inline std::size_t tensorTypeSize(rknn_tensor_type type)
        {
            switch (type)
            {
            case RKNN_TENSOR_FLOAT32:
                return sizeof(float);
            case RKNN_TENSOR_FLOAT16:
                return sizeof(uint16_t);
            case RKNN_TENSOR_INT8:
                return sizeof(int8_t);
            case RKNN_TENSOR_UINT8:
                return sizeof(uint8_t);
            case RKNN_TENSOR_INT16:
                return sizeof(int16_t);
            case RKNN_TENSOR_UINT16:
                return sizeof(uint16_t);
            default:
                throw std::runtime_error("Unsupported tensor type");
            }
        }

        inline void resolveSpatial(const rknn_tensor_attr &attr, int &height, int &width)
        {
            if (attr.fmt == RKNN_TENSOR_NCHW)
            {
                height = attr.dims[2];
                width = attr.dims[3];
            }
            else
            {
                height = attr.dims[1];
                width = attr.dims[2];
            }
        }

        template <typename T>
        inline T clampValue(T v, T minVal, T maxVal)
        {
            return std::max(minVal, std::min(maxVal, v));
        }
    }

#define RKNN_CHECK(op, msg)                                                                                              \
    do                                                                                                                   \
    {                                                                                                                    \
        int ret = (op);                                                                                                  \
        if (ret < 0)                                                                                                     \
        {                                                                                                                \
            LOG.error("[RkRunner] %s failed with ret=%d", msg, ret);                                                     \
            throw std::runtime_error(msg);                                                                               \
        }                                                                                                                \
    } while (0)

    RkRunner::RkRunner(std::string_view modelPath, int npuId)
    {
        auto modelData = loadModel(modelPath);
        if (!modelData)
        {
            LOG.error("[RkRunner] Failed to load model from %s", std::string(modelPath).c_str());
            throw std::runtime_error("Failed to load model");
        }

        auto &[buffer, length] = *modelData;
        int initRet = rknn_init(&ctx_, buffer.get(), length, 0, nullptr);
        if (initRet < 0)
        {
            LOG.error("[RkRunner] rknn_init failed with ret=%d. Please confirm rknpu driver is loaded (insmod rknpu).", initRet);
            throw std::runtime_error("rknn_init");
        }

        if (npuId >= 0 && npuId < static_cast<int>(std::size(kCoreMask)))
        {
            RKNN_CHECK(rknn_set_core_mask(ctx_, kCoreMask[npuId]), "rknn_set_core_mask");
        }

        prepareTensorAttributes();
        initialized_ = true;
    }

    RkRunner::~RkRunner()
    {
        releaseResources();
    }

    std::optional<RkRunner::ModelData> RkRunner::loadModel(std::string_view path)
    {
        std::ifstream fin(path.data(), std::ios::binary | std::ios::ate);
        if (!fin.is_open())
        {
            return std::nullopt;
        }

        std::size_t fileSize = static_cast<std::size_t>(fin.tellg());
        fin.seekg(0, std::ios::beg);
        auto buffer = std::make_unique<unsigned char[]>(fileSize);
        fin.read(reinterpret_cast<char *>(buffer.get()), fileSize);
        return ModelData{std::move(buffer), fileSize};
    }

    void RkRunner::dumpTensorAttr(const rknn_tensor_attr &attr)
    {
        LOG.info("[RkRunner] index=%d, name=%s, dims=[%d,%d,%d,%d], size=%d, type=%s, qnt_type=%s, zp=%d, scale=%f",
                 attr.index,
                 attr.name,
                 attr.dims[0],
                 attr.dims[1],
                 attr.dims[2],
                 attr.dims[3],
                 attr.size,
                 get_type_string(attr.type),
                 get_qnt_type_string(attr.qnt_type),
                 attr.zp,
                 attr.scale);
    }

    void RkRunner::prepareTensorAttributes()
    {
        rknn_input_output_num ioNum{};
        RKNN_CHECK(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "rknn_query_in_out_num");

        inputAttrs_.resize(ioNum.n_input);
        for (std::size_t i = 0; i < inputAttrs_.size(); ++i)
        {
            auto &attr = inputAttrs_[i];
            memset(&attr, 0, sizeof(rknn_tensor_attr));
            attr.index = static_cast<int>(i);
            RKNN_CHECK(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "rknn_query_input_attr");
            dumpTensorAttr(attr);

            auto *tensorMem = rknn_create_mem(ctx_, attr.size);
            if (!tensorMem)
            {
                throw std::runtime_error("Failed to allocate input tensor memory");
            }
            inputMems_.push_back(tensorMem);
            RKNN_CHECK(rknn_set_io_mem(ctx_, tensorMem, &attr), "rknn_set_input_mem");
        }

        outputAttrs_.resize(ioNum.n_output);
        for (std::size_t i = 0; i < outputAttrs_.size(); ++i)
        {
            auto &attr = outputAttrs_[i];
            memset(&attr, 0, sizeof(rknn_tensor_attr));
            attr.index = static_cast<int>(i);
            RKNN_CHECK(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "rknn_query_output_attr");
            dumpTensorAttr(attr);

            // Force float32 dump for float16
            if (attr.type == RKNN_TENSOR_FLOAT16)
            {
                attr.type = RKNN_TENSOR_FLOAT32;
            }

            std::size_t bytes = attr.n_elems * tensorTypeSize(attr.type);
            auto *tensorMem = rknn_create_mem(ctx_, bytes);
            if (!tensorMem)
            {
                throw std::runtime_error("Failed to allocate output tensor memory");
            }
            outputMems_.push_back(tensorMem);
            RKNN_CHECK(rknn_set_io_mem(ctx_, tensorMem, &attr), "rknn_set_output_mem");
        }
    }

    void RkRunner::releaseResources()
    {
        for (auto *mem : inputMems_)
        {
            if (mem)
            {
                rknn_destroy_mem(ctx_, mem);
            }
        }
        inputMems_.clear();

        for (auto *mem : outputMems_)
        {
            if (mem)
            {
                rknn_destroy_mem(ctx_, mem);
            }
        }
        outputMems_.clear();

        if (initialized_)
        {
            rknn_destroy(ctx_);
            initialized_ = false;
        }
    }

    void RkRunner::copyInputData(const cv::Mat &mat, std::size_t index)
    {
        if (index >= inputAttrs_.size() || index >= inputMems_.size())
        {
            throw std::out_of_range("Input index out of range");
        }

        auto &attr = inputAttrs_[index];
        auto *tensorMem = inputMems_[index];
        std::size_t elemCount = attr.n_elems;

        cv::Mat floatMat;
        if (mat.type() != CV_32FC1)
        {
            mat.convertTo(floatMat, CV_32FC1);
        }
        else
        {
            floatMat = mat;
        }

        if (static_cast<std::size_t>(floatMat.total()) != elemCount)
        {
            LOG.error("[RkRunner] Input element count mismatch: expected %zu, got %zu", elemCount, floatMat.total());
            throw std::runtime_error("Input size mismatch");
        }

        const float *src = floatMat.ptr<float>();

        switch (attr.type)
        {
        case RKNN_TENSOR_FLOAT32:
            std::memcpy(tensorMem->virt_addr, src, elemCount * sizeof(float));
            break;
        case RKNN_TENSOR_INT8:
        {
            auto *dst = reinterpret_cast<int8_t *>(tensorMem->virt_addr);
            float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
            for (std::size_t i = 0; i < elemCount; ++i)
            {
                float quant = std::round(src[i] / scale) + attr.zp;
                dst[i] = static_cast<int8_t>(clampValue<int>(static_cast<int>(quant), -128, 127));
            }
            break;
        }
        case RKNN_TENSOR_UINT8:
        {
            auto *dst = reinterpret_cast<uint8_t *>(tensorMem->virt_addr);
            float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
            for (std::size_t i = 0; i < elemCount; ++i)
            {
                float quant = std::round(src[i] / scale) + attr.zp;
                dst[i] = static_cast<uint8_t>(clampValue<int>(static_cast<int>(quant), 0, 255));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported input tensor type");
        }

        RKNN_CHECK(rknn_mem_sync(ctx_, tensorMem, RKNN_MEMORY_SYNC_TO_DEVICE), "rknn_mem_sync_input");
    }

    cv::Mat RkRunner::fetchOutputData(std::size_t index)
    {
        if (index >= outputAttrs_.size() || index >= outputMems_.size())
        {
            throw std::out_of_range("Output index out of range");
        }

        auto &attr = outputAttrs_[index];
        auto *tensorMem = outputMems_[index];
        std::size_t elemCount = attr.n_elems;

        RKNN_CHECK(rknn_mem_sync(ctx_, tensorMem, RKNN_MEMORY_SYNC_FROM_DEVICE), "rknn_mem_sync_output");

        int height = 0;
        int width = 0;
        resolveSpatial(attr, height, width);

        cv::Mat result(height, width, CV_32FC1);
        float *dst = result.ptr<float>();

        switch (attr.type)
        {
        case RKNN_TENSOR_FLOAT32:
            std::memcpy(dst, tensorMem->virt_addr, elemCount * sizeof(float));
            break;
        case RKNN_TENSOR_INT8:
        {
            auto *src = reinterpret_cast<int8_t *>(tensorMem->virt_addr);
            float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
            for (std::size_t i = 0; i < elemCount; ++i)
            {
                dst[i] = scale * (static_cast<int32_t>(src[i]) - attr.zp);
            }
            break;
        }
        case RKNN_TENSOR_UINT8:
        {
            auto *src = reinterpret_cast<uint8_t *>(tensorMem->virt_addr);
            float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
            for (std::size_t i = 0; i < elemCount; ++i)
            {
                dst[i] = scale * (static_cast<int32_t>(src[i]) - attr.zp);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported output tensor type");
        }

        return result;
    }

    std::shared_ptr<DataObject> RkRunner::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        if (!initialized_)
        {
            LOG.error("[RkRunner] Runner not initialized");
            return nullptr;
        }

        if (inputs.size() != 1)
        {
            LOG.error("[RkRunner] Expected 1 input, got %zu", inputs.size());
            return nullptr;
        }

        auto prepPackage = std::dynamic_pointer_cast<FusionPreprocessPackage>(inputs[0]);
        if (!prepPackage)
        {
            LOG.error("[RkRunner] Invalid preprocess package");
            return nullptr;
        }

        copyInputData(prepPackage->get_vis_y(), 0);
        if (inputAttrs_.size() > 1)
        {
            copyInputData(prepPackage->get_infrared(), 1);
        }

        RKNN_CHECK(rknn_run(ctx_, nullptr), "rknn_run");

        cv::Mat fusedY = fetchOutputData(0);
        return std::make_shared<FusionRunnerPackage>(fusedY, prepPackage->get_id());
    }
}
