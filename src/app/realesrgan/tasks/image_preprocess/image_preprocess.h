#pragma once

#include "framework/processing_task.h"

namespace GryFlux {
namespace RealESRGAN {

class ImagePreprocess : public GryFlux::ProcessingTask {
public:
    ImagePreprocess(int model_width, int model_height, int pad_value = 0)
        : model_width_(model_width), model_height_(model_height), pad_value_(pad_value) {}

    std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

private:
    int model_width_;
    int model_height_;
    int pad_value_;
};

} // namespace RealESRGAN
} // namespace GryFlux
