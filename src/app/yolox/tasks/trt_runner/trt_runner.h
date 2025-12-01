#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "framework/processing_task.h"
#include "utils/logger.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"

namespace GryFlux
{
    using EngineData = std::pair<std::unique_ptr<char[]>, std::size_t>; // buffer, size

    // TensorRT logger
    class TRTLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override
        {
            // Filter out info messages
            if (severity <= Severity::kWARNING)
                LOG.info("[TensorRT] %s", msg);
        }
    };

    class TrtRunner : public ProcessingTask
    {
    public:
        explicit TrtRunner(std::string_view engine_path, 
                          const std::size_t model_width = 640, 
                          const std::size_t model_height = 640);
        
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

        // Function to read engine file into a buffer
        std::optional<EngineData> load_engine(std::string_view filename);

        ~TrtRunner();

    private:
        bool is_supported_dtype(nvinfer1::DataType type) const;
        std::size_t element_size(nvinfer1::DataType type) const;
        const void* prepare_input_buffer();
        std::shared_ptr<float[]> fetch_output_as_float(int output_index);

        TRTLogger logger_;
        nvinfer1::IRuntime* runtime_;
        nvinfer1::ICudaEngine* engine_;
        nvinfer1::IExecutionContext* context_;
        
        cudaStream_t stream_;
        
        // Model dimensions
        std::size_t model_width_;
        std::size_t model_height_;
        
        // Input/Output info
        std::string input_name_;
        std::string output_name_;
        std::size_t input_size_;
        std::size_t output_size_;
        nvinfer1::DataType input_data_type_;
        std::vector<nvinfer1::DataType> output_data_types_;
        
        // Device buffers
        void* input_buffer_;
        std::vector<void*> output_buffers_;
        std::vector<std::size_t> output_sizes_;
        std::vector<uint8_t> input_converted_buffer_;
        std::vector<std::vector<uint8_t>> host_output_buffers_;
        
        // Number of outputs
        int num_outputs_;
        
        // Output dimensions for each output layer
        std::vector<std::vector<int>> output_dims_;
        
        // Host buffer for preprocessed input (CHW format, float)
        std::vector<float> host_input_buffer_;
        
        // Initialize CUDA and TensorRT resources
        void initializeTensorRT(std::string_view engine_path);
        void allocateBuffers();
        void cleanupBuffers();
        
        // Preprocess image: convert HWC uint8 to CHW float
        void preprocess_image(const cv::Mat& frame, float* output_buffer);
    };
}
