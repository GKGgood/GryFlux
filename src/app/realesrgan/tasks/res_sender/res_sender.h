#pragma once

#include "framework/processing_task.h"

namespace GryFlux {
namespace RealESRGAN {

class ResSender : public GryFlux::ProcessingTask {
public:
    ResSender() = default;
    std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;
};

} // namespace RealESRGAN
} // namespace GryFlux
