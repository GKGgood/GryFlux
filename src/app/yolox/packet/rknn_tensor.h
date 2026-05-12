#pragma once

#include <cstddef>
#include <vector>

namespace GryFlux
{

struct RknnTensor
{
    std::vector<float> data;
    std::size_t channels = 0;
    std::size_t gridHeight = 0;
    std::size_t gridWidth = 0;
};

} // namespace GryFlux
