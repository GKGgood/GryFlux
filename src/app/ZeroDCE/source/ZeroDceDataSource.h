#pragma once

#include "framework/data_source.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "No filesystem support found"
#endif

#include <string>
#include <vector>

class ZeroDceDataSource : public GryFlux::DataSource
{
public:
    explicit ZeroDceDataSource(const std::string &inputDir,
                               int inputWidth,
                               int inputHeight);

    std::size_t GetTotalFrames() const
    {
        return image_files_.size();
    }

    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    static bool IsImageFile(const fs::path &path);

    fs::path input_dir_;
    std::vector<fs::path> image_files_;
    std::size_t cursor_ = 0;
    int idx_ = 0;
    int input_width_ = 0;
    int input_height_ = 0;
};
