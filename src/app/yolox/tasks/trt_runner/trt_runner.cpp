#include "trt_runner.h"
#include "package.h"
#include "utils/logger.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = call;                                              \
        if (status != cudaSuccess) {                                            \
            LOG.error("CUDA error: %s at %s:%d", cudaGetErrorString(status),    \
                     __FILE__, __LINE__);                                       \
            throw std::runtime_error("CUDA error");                             \
        }                                                                       \
    } while (0)

namespace GryFlux {

namespace {
    const char* dtype_to_string(nvinfer1::DataType type)
    {
        switch (type) {
            case nvinfer1::DataType::kFLOAT: return "FP32";
            case nvinfer1::DataType::kHALF:  return "FP16";
            case nvinfer1::DataType::kINT8:  return "INT8";
            default:                         return "Unsupported";
        }
    }
}

        TrtRunner::TrtRunner(std::string_view engine_path, std::size_t model_width, std::size_t model_height)
                : runtime_(nullptr), engine_(nullptr), context_(nullptr),
                    stream_(nullptr), model_width_(model_width), model_height_(model_height),
                    input_size_(0), output_size_(0), input_data_type_(nvinfer1::DataType::kFLOAT),
                    input_buffer_(nullptr), num_outputs_(0)
    {
        LOG.info("Initializing TensorRT runner with engine: %s", engine_path.data());
        initializeTensorRT(engine_path);
        allocateBuffers();
        LOG.info("TensorRT runner initialized successfully");
    }

    void TrtRunner::initializeTensorRT(std::string_view engine_path)
    {
        // Load engine file
        auto engine_data = load_engine(engine_path);
        if (!engine_data) {
            LOG.error("Failed to load engine file: %s", engine_path.data());
            throw std::runtime_error("Failed to load engine file");
        }

        auto& [engine_buffer, engine_size] = *engine_data;
        LOG.info("Engine file loaded, size: %zu bytes", engine_size);

        // Create TensorRT runtime
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) {
            LOG.error("Failed to create TensorRT runtime");
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        // Deserialize engine
        engine_ = runtime_->deserializeCudaEngine(engine_buffer.get(), engine_size);
        if (!engine_) {
            LOG.error("Failed to deserialize CUDA engine");
            throw std::runtime_error("Failed to deserialize CUDA engine");
        }

        // Create execution context
        context_ = engine_->createExecutionContext();
        if (!context_) {
            LOG.error("Failed to create execution context");
            throw std::runtime_error("Failed to create execution context");
        }

        // Create CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream_));

        // Get input/output information
        int num_bindings = engine_->getNbBindings();
        LOG.info("Number of bindings: %d", num_bindings);

        for (int i = 0; i < num_bindings; ++i) {
            const char* name = engine_->getBindingName(i);
            auto dims = engine_->getBindingDimensions(i);
            bool is_input = engine_->bindingIsInput(i);
            auto dtype = engine_->getBindingDataType(i);

            if (!is_supported_dtype(dtype)) {
                LOG.error("Binding %s uses unsupported data type", name);
                throw std::runtime_error("Unsupported TensorRT binding data type");
            }
            
            std::string dims_str = "[";
            for (int j = 0; j < dims.nbDims; ++j) {
                dims_str += std::to_string(dims.d[j]);
                if (j < dims.nbDims - 1) dims_str += ", ";
            }
            dims_str += "]";
            
            LOG.info("Binding %d: %s, is_input: %d, dims: %s, dtype: %s",
                     i, name, is_input, dims_str.c_str(), dtype_to_string(dtype));

            if (is_input) {
                input_name_ = name;
                // Calculate input size (assuming NCHW or NHWC format)
                input_size_ = 1;
                for (int j = 0; j < dims.nbDims; ++j) {
                    input_size_ *= dims.d[j];
                }
                input_data_type_ = dtype;
                LOG.info("Input size: %zu", input_size_);
            } else {
                // Output binding
                num_outputs_++;
                std::size_t output_size = 1;
                std::vector<int> output_dim;
                for (int j = 0; j < dims.nbDims; ++j) {
                    output_size *= dims.d[j];
                    output_dim.push_back(dims.d[j]);
                }
                output_sizes_.push_back(output_size);
                output_dims_.push_back(output_dim);
                output_data_types_.push_back(dtype);
                LOG.info("Output %d size: %zu", num_outputs_ - 1, output_size);
            }
        }

        host_output_buffers_.resize(output_sizes_.size());

        LOG.info("TensorRT initialization completed. Input: %s (%s), Outputs: %d", 
                 input_name_.c_str(), dtype_to_string(input_data_type_), num_outputs_);
    }

    void TrtRunner::allocateBuffers()
    {
        if (input_size_ == 0) {
            throw std::runtime_error("Input binding size is zero; cannot allocate buffers");
        }
        const std::size_t input_bytes = input_size_ * element_size(input_data_type_);
        CUDA_CHECK(cudaMalloc(&input_buffer_, input_bytes));
        LOG.info("Allocated input buffer: %zu bytes (%s)", input_bytes, dtype_to_string(input_data_type_));

        // Pre-allocate host input buffer in float (preprocess domain)
        host_input_buffer_.resize(input_size_);
        LOG.info("Pre-allocated host input buffer: %zu elements (%zu bytes FP32)",
                 host_input_buffer_.size(), host_input_buffer_.size() * sizeof(float));

        if (input_data_type_ != nvinfer1::DataType::kFLOAT) {
            input_converted_buffer_.resize(input_bytes);
        } else {
            input_converted_buffer_.clear();
        }

        // Allocate output buffers
        for (std::size_t i = 0; i < output_sizes_.size(); ++i) {
            void* output_buffer = nullptr;
            const auto dtype = output_data_types_[i];
            const std::size_t output_bytes = output_sizes_[i] * element_size(dtype);
            CUDA_CHECK(cudaMalloc(&output_buffer, output_bytes));
            output_buffers_.push_back(output_buffer);
            LOG.info("Allocated output buffer %zu: %zu bytes (%s)", i, output_bytes, dtype_to_string(dtype));

            if (dtype != nvinfer1::DataType::kFLOAT) {
                host_output_buffers_[i].resize(output_bytes);
            } else {
                host_output_buffers_[i].clear();
            }
        }
    }

    void TrtRunner::cleanupBuffers()
    {
        if (input_buffer_) {
            cudaFree(input_buffer_);
            input_buffer_ = nullptr;
        }

        for (auto& buffer : output_buffers_) {
            if (buffer) {
                cudaFree(buffer);
            }
        }
        output_buffers_.clear();
        host_output_buffers_.clear();
        input_converted_buffer_.clear();
        output_sizes_.clear();
        output_dims_.clear();
        output_data_types_.clear();
        input_size_ = 0;
        num_outputs_ = 0;
    }

    std::optional<EngineData> TrtRunner::load_engine(std::string_view filename)
    {
        std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
        
        if (!file.is_open()) {
            LOG.error("Failed to open engine file: %s", filename.data());
            return std::nullopt;
        }

        const std::size_t fileSize = static_cast<std::size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        
        auto buffer = std::make_unique<char[]>(fileSize);
        if (!buffer) {
            LOG.error("Memory allocation failed for engine buffer");
            return std::nullopt;
        }

        file.read(buffer.get(), fileSize);
        file.close();
        
        return std::pair{std::move(buffer), fileSize};
    }

    void TrtRunner::preprocess_image(const cv::Mat& frame, float* output_buffer)
    {
        const int channels = 3;
        const size_t channel_size = model_width_ * model_height_;
        
        // Convert HWC uint8 to CHW float in a single loop
        if (frame.type() == CV_8UC3) {
            // Most common case: 8-bit RGB image
            const uint8_t* src = frame.data;
            for (size_t h = 0; h < model_height_; ++h) {
                for (size_t w = 0; w < model_width_; ++w) {
                    size_t src_idx = (h * model_width_ + w) * channels;
                    size_t dst_idx = h * model_width_ + w;
                    // HWC to CHW (RGB order)
                    output_buffer[dst_idx] = static_cast<float>(src[src_idx]);                    // R
                    output_buffer[channel_size + dst_idx] = static_cast<float>(src[src_idx + 1]); // G
                    output_buffer[2 * channel_size + dst_idx] = static_cast<float>(src[src_idx + 2]); // B
                }
            }
        } else if (frame.type() == CV_32FC3) {
            // Already float type
            const float* src = reinterpret_cast<const float*>(frame.data);
            for (size_t h = 0; h < model_height_; ++h) {
                for (size_t w = 0; w < model_width_; ++w) {
                    size_t src_idx = (h * model_width_ + w) * channels;
                    size_t dst_idx = h * model_width_ + w;
                    output_buffer[dst_idx] = src[src_idx];                    // R
                    output_buffer[channel_size + dst_idx] = src[src_idx + 1]; // G
                    output_buffer[2 * channel_size + dst_idx] = src[src_idx + 2]; // B
                }
            }
        } else {
            // Fallback for other types
            cv::Mat float_frame;
            frame.convertTo(float_frame, CV_32F);
            
            std::vector<cv::Mat> channels_vec(channels);
            cv::split(float_frame, channels_vec);
            
            for (int i = 0; i < channels; i++) {
                memcpy(output_buffer + i * channel_size, channels_vec[i].data, channel_size * sizeof(float));
            }
        }
    }

    bool TrtRunner::is_supported_dtype(nvinfer1::DataType type) const
    {
        return type == nvinfer1::DataType::kFLOAT ||
               type == nvinfer1::DataType::kHALF ||
               type == nvinfer1::DataType::kINT8;
    }

    std::size_t TrtRunner::element_size(nvinfer1::DataType type) const
    {
        switch (type) {
            case nvinfer1::DataType::kFLOAT: return sizeof(float);
            case nvinfer1::DataType::kHALF:  return sizeof(uint16_t);
            case nvinfer1::DataType::kINT8:  return sizeof(int8_t);
            default:
                LOG.error("Unsupported TensorRT data type encountered");
                throw std::runtime_error("Unsupported TensorRT data type");
        }
    }

    const void* TrtRunner::prepare_input_buffer()
    {
        switch (input_data_type_) {
            case nvinfer1::DataType::kFLOAT:
                return host_input_buffer_.data();
            case nvinfer1::DataType::kHALF: {
                auto dst = reinterpret_cast<__half*>(input_converted_buffer_.data());
                for (std::size_t idx = 0; idx < input_size_; ++idx) {
                    dst[idx] = __float2half(host_input_buffer_[idx]);
                }
                return dst;
            }
            case nvinfer1::DataType::kINT8: {
                auto dst = reinterpret_cast<int8_t*>(input_converted_buffer_.data());
                for (std::size_t idx = 0; idx < input_size_; ++idx) {
                    float centered = host_input_buffer_[idx] - 128.0f;
                    centered = std::clamp(std::round(centered), -128.0f, 127.0f);
                    dst[idx] = static_cast<int8_t>(centered);
                }
                return dst;
            }
            default:
                LOG.error("Unexpected input data type");
                throw std::runtime_error("Unexpected input data type");
        }
    }

    std::shared_ptr<float[]> TrtRunner::fetch_output_as_float(int output_index)
    {
        if (output_index < 0 || output_index >= num_outputs_) {
            LOG.error("Output index %d out of range", output_index);
            return nullptr;
        }

        const auto dtype = output_data_types_[output_index];
        const auto elem_count = output_sizes_[output_index];
        const std::size_t bytes = elem_count * element_size(dtype);
        auto result = std::shared_ptr<float[]>(new float[elem_count]);

        if (dtype == nvinfer1::DataType::kFLOAT) {
            CUDA_CHECK(cudaMemcpyAsync(result.get(),
                                      output_buffers_[output_index],
                                      bytes,
                                      cudaMemcpyDeviceToHost,
                                      stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            return result;
        }

        auto &staging = host_output_buffers_[output_index];
        if (staging.size() != bytes) {
            staging.resize(bytes);
        }

        CUDA_CHECK(cudaMemcpyAsync(staging.data(),
                                   output_buffers_[output_index],
                                   bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (dtype == nvinfer1::DataType::kHALF) {
            const __half* src = reinterpret_cast<const __half*>(staging.data());
            for (std::size_t i = 0; i < elem_count; ++i) {
                result[i] = __half2float(src[i]);
            }
            return result;
        }

        if (dtype == nvinfer1::DataType::kINT8) {
            const int8_t* src = reinterpret_cast<const int8_t*>(staging.data());
            for (std::size_t i = 0; i < elem_count; ++i) {
                result[i] = static_cast<float>(src[i]);
            }
            return result;
        }

        LOG.error("Unsupported output data type detected");
        return nullptr;
    }

    std::shared_ptr<DataObject> TrtRunner::process(const std::vector<std::shared_ptr<DataObject>>& inputs)
    {
        if (inputs.size() != 1) {
            LOG.error("TrtRunner expects exactly 1 input, got %zu", inputs.size());
            return nullptr;
        }

        auto input_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        if (!input_data) {
            LOG.error("Failed to cast input to ImagePackage");
            return nullptr;
        }

        auto frame = input_data->get_data();
        
        // Validate input dimensions
        if (frame.rows != static_cast<int>(model_height_) || frame.cols != static_cast<int>(model_width_)) {
            LOG.error("Input image size (%dx%d) doesn't match model input size (%zux%zu)", 
                      frame.cols, frame.rows, model_width_, model_height_);
            return nullptr;
        }

        // Preprocess: convert HWC uint8 to CHW float
        preprocess_image(frame, host_input_buffer_.data());

        // Copy preprocessed input data to device
        const void* upload_ptr = prepare_input_buffer();
        const size_t input_bytes = input_size_ * element_size(input_data_type_);
        CUDA_CHECK(cudaMemcpyAsync(input_buffer_, upload_ptr,
                       input_bytes,
                       cudaMemcpyHostToDevice, stream_));

        // Prepare bindings
        std::vector<void*> bindings(1 + num_outputs_);
        bindings[0] = input_buffer_; // Input binding
        for (int i = 0; i < num_outputs_; ++i) {
            bindings[i + 1] = output_buffers_[i]; // Output bindings
        }

        // Execute inference
        bool status = context_->enqueueV2(bindings.data(), stream_, nullptr);
        if (!status) {
            LOG.error("TensorRT inference execution failed");
            return nullptr;
        }

        // Create output package
        auto output_data = std::make_shared<RunnerPackage>(model_width_, model_height_);

        // Copy outputs from device to host
        for (int i = 0; i < num_outputs_; ++i) {
            auto output = fetch_output_as_float(i);
            if (!output) {
                LOG.error("Failed to fetch output %d", i);
                return nullptr;
            }

            // Calculate grid dimensions
            // Assuming output format is [batch, num_anchors, grid_h, grid_w] or similar
            // For YOLOX, typical output is [1, 85, grid_h, grid_w] where 85 = 5 + num_classes
            std::size_t grid_h = 1, grid_w = 1;
            if (output_dims_[i].size() >= 3) {
                grid_h = output_dims_[i][output_dims_[i].size() - 2];
                grid_w = output_dims_[i][output_dims_[i].size() - 1];
            }

            LOG.info("Output %d: grid_h=%zu, grid_w=%zu, total_elements=%zu", 
                     i, grid_h, grid_w, output_sizes_[i]);

            output_data->push_data({output, output_sizes_[i]}, {grid_h, grid_w});
        }

        return output_data;
    }

    TrtRunner::~TrtRunner()
    {
        LOG.info("Destroying TrtRunner");
        
        cleanupBuffers();
        
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        
        if (context_) {
            context_->destroy();
            context_ = nullptr;
        }
        
        if (engine_) {
            engine_->destroy();
            engine_ = nullptr;
        }
        
        if (runtime_) {
            runtime_->destroy();
            runtime_ = nullptr;
        }
        
        LOG.info("TrtRunner destroyed");
    }

} // namespace GryFlux
