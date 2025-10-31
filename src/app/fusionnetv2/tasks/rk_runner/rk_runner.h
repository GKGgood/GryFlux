#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include <opencv2/opencv.hpp>

#include "framework/processing_task.h"
#include "rknn_api.h"

namespace GryFlux
{
    class RkRunner : public ProcessingTask
    {
    public:
        explicit RkRunner(std::string_view modelPath, int npuId = 1);
        ~RkRunner();

        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

    private:
        using ModelData = std::pair<std::unique_ptr<unsigned char[]>, std::size_t>;

        rknn_context ctx_;
        std::vector<rknn_tensor_attr> inputAttrs_;
        std::vector<rknn_tensor_attr> outputAttrs_;
        std::vector<rknn_tensor_mem *> inputMems_;
        std::vector<rknn_tensor_mem *> outputMems_;

        std::optional<ModelData> loadModel(std::string_view path);
        static void dumpTensorAttr(const rknn_tensor_attr &attr);
        void prepareTensorAttributes();
        void releaseResources();
        void copyInputData(const cv::Mat &mat, std::size_t index);
    cv::Mat fetchOutputData(std::size_t index);

        bool initialized_{false};
    };
}
