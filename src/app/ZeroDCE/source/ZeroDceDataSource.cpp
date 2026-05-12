#include "source/ZeroDceDataSource.h"

#include "packet/zero_dce_packet.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

ZeroDceDataSource::ZeroDceDataSource(const std::string &inputDir,
                                     int inputWidth,
                                     int inputHeight)
    : input_dir_(inputDir),
      input_width_(inputWidth),
      input_height_(inputHeight)
{
    if (!fs::exists(input_dir_) || !fs::is_directory(input_dir_))
    {
        throw std::runtime_error("ZeroDCE input_dir does not exist or is not a directory: " + inputDir);
    }

    for (const auto &entry : fs::directory_iterator(input_dir_))
    {
        if (!fs::is_regular_file(entry.status()))
        {
            continue;
        }
        if (!IsImageFile(entry.path()))
        {
            continue;
        }
        image_files_.push_back(entry.path());
    }

    std::sort(image_files_.begin(), image_files_.end());
    setHasMore(!image_files_.empty());
}

std::unique_ptr<GryFlux::DataPacket> ZeroDceDataSource::produce()
{
    while (cursor_ < image_files_.size())
    {
        const auto imagePath = image_files_[cursor_++];
        cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            continue;
        }

        auto packet = std::make_unique<ZeroDcePacket>(input_width_, input_height_);
        packet->frame_id = static_cast<uint64_t>(idx_++);
        packet->output_relative_path = imagePath.filename().string();
        packet->input_image = image;

        if (cursor_ >= image_files_.size())
        {
            setHasMore(false);
        }

        return packet;
    }

    setHasMore(false);
    return nullptr;
}

bool ZeroDceDataSource::IsImageFile(const fs::path &path)
{
    if (!path.has_extension())
    {
        return false;
    }

    std::string ext = path.extension().string();
    std::transform(
        ext.begin(),
        ext.end(),
        ext.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(std::tolower(c));
        });

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}
