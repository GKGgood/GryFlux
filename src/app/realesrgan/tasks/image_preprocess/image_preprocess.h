#pragma once

#include "framework/processing_task.h"

namespace GryFlux {
namespace RealESRGAN {

class ImagePreprocess : public GryFlux::ProcessingTask {
public:
    ImagePreprocess(int model_width, int model_height)
        : model_width_(model_width), model_height_(model_height) {}

    std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

private:
    int model_width_;
    int model_height_;
};

} // namespace RealESRGAN
} // namespace GryFlux
