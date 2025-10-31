#pragma once

#include "framework/processing_task.h"

namespace GryFlux
{
    class ImagePreprocess : public ProcessingTask
    {
    public:
        ImagePreprocess(int modelWidth, int modelHeight);
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

    private:
        int modelWidth_;
        int modelHeight_;
    };
}
