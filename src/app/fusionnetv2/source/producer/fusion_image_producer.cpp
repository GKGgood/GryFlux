#include "fusion_image_producer.h"

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "package.h"

namespace GryFlux
{
    namespace
    {
        constexpr const char *kVisibleFolder = "visible";
        constexpr const char *kInfraredFolder = "infrared";
    }

    bool FusionImageProducer::isImageFile(const std::filesystem::path &path)
    {
        static const std::vector<std::string> kExtensions = {".jpg", ".jpeg", ".png", ".bmp"};
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end();
    }

    FusionImageProducer::FusionImageProducer(StreamingPipeline &pipeline,
                                             std::atomic<bool> &running,
                                             CPUAllocator *allocator,
                                             std::string_view datasetRoot,
                                             std::size_t maxFrames)
        : DataProducer(pipeline, running, allocator),
          maxFrames_(maxFrames),
          frameCount_(0)
    {
        std::filesystem::path root(datasetRoot);
        if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root))
        {
            LOG.error("[FusionImageProducer] Dataset root %s is invalid", root.string().c_str());
            throw std::runtime_error("Invalid dataset root path");
        }

        visiblePath_ = root / kVisibleFolder;
        infraredPath_ = root / kInfraredFolder;

        if (!std::filesystem::exists(visiblePath_) || !std::filesystem::is_directory(visiblePath_))
        {
            LOG.error("[FusionImageProducer] Visible folder %s is invalid", visiblePath_.string().c_str());
            throw std::runtime_error("Missing visible folder");
        }

        if (!std::filesystem::exists(infraredPath_) || !std::filesystem::is_directory(infraredPath_))
        {
            LOG.error("[FusionImageProducer] Infrared folder %s is invalid", infraredPath_.string().c_str());
            throw std::runtime_error("Missing infrared folder");
        }

        for (auto &entry : std::filesystem::directory_iterator(visiblePath_))
        {
            if (!entry.is_regular_file() || !isImageFile(entry.path()))
            {
                continue;
            }

            auto filename = entry.path().filename().string();
            auto infraredFile = infraredPath_ / filename;
            if (!std::filesystem::exists(infraredFile))
            {
                LOG.warning("[FusionImageProducer] Infrared image %s not found, skipping", infraredFile.string().c_str());
                continue;
            }

            fileList_.push_back(filename);
        }

        std::sort(fileList_.begin(), fileList_.end());

        if (fileList_.empty())
        {
            LOG.error("[FusionImageProducer] No image pairs found under %s", root.string().c_str());
            throw std::runtime_error("No valid image pairs found");
        }
    }

    FusionImageProducer::~FusionImageProducer() = default;

    void FusionImageProducer::run()
    {
        LOG.info("[FusionImageProducer] Producer started, total pairs: %zu", fileList_.size());

        for (const auto &filename : fileList_)
        {
            if (!running.load())
            {
                break;
            }

            if (maxFrames_ != static_cast<std::size_t>(-1) && static_cast<std::size_t>(frameCount_) >= maxFrames_)
            {
                break;
            }

            auto visPath = visiblePath_ / filename;
            auto irPath = infraredPath_ / filename;

            cv::Mat visible = cv::imread(visPath.string(), cv::IMREAD_COLOR);
            cv::Mat infrared = cv::imread(irPath.string(), cv::IMREAD_GRAYSCALE);

            if (visible.empty() || infrared.empty())
            {
                LOG.error("[FusionImageProducer] Failed to read pair (%s, %s)", visPath.string().c_str(), irPath.string().c_str());
                continue;
            }

            auto package = std::make_shared<FusionImagePackage>(visible, infrared, frameCount_);
            if (!addData(package))
            {
                LOG.error("[FusionImageProducer] Failed to enqueue data for frame %d", frameCount_);
                break;
            }

            ++frameCount_;
        }

        LOG.info("[FusionImageProducer] Producer finished, generated %d frames", frameCount_);
        this->stop();
    }
}
