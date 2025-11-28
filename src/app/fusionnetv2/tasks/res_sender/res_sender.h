#pragma once

#include "framework/processing_task.h"

namespace GryFlux
{
    class ResSender : public ProcessingTask
    {
    public:
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;
    };
}
