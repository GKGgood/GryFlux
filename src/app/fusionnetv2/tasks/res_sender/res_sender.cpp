#include "res_sender.h"

#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    std::shared_ptr<DataObject> ResSender::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        if (inputs.size() != 1)
        {
            LOG.error("[ResSender] Expected 1 input, got %zu", inputs.size());
            return nullptr;
        }

        auto result = std::dynamic_pointer_cast<FusionResultPackage>(inputs[0]);
        if (!result)
        {
            LOG.error("[ResSender] Invalid result package");
            return nullptr;
        }

        return std::make_shared<FusionResultPackage>(result->get_result(), result->get_id());
    }
}
