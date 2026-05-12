#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/result_consumer.h"
#include "context/infercontext.h"
#include "context/reid_context.h"
#include "nodes/yolox_deepsort_nodes.h"
#include "source/video_source.h"
#include "utils/datatype.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct CliOptions
{
    int classCount = 80;
    int reidFeatureDim = kFeatureDim;
    int deviceId = 0;
    float confThreshold = 0.3f;
    float nmsThreshold = 0.45f;
    std::size_t maxDetections = 100;
    std::size_t detectionNpuInstances = 3;
    std::size_t reidNpuInstances = 3;
    std::size_t threadPoolSize = 8;
    std::size_t maxActivePackets = 4;
    bool enableProfiling = true;
};

void printHelp()
{
    LOG.info("Usage: yolox_deepsort <yolox_model> <reid_model> <input_video> [output_video]");
    LOG.info("Pipeline config is defined in CliOptions in src/app/yolox_deepsort/yolox_deepsort.cpp");
}
} // namespace

int main(int argc, char **argv)
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("yolox_deepsort");

    if (argc < 4 || argc > 5)
    {
        printHelp();
        return -1;
    }

    const std::string yoloxModelPath = argv[1];
    const std::string reidModelPath = argv[2];
    const std::string inputPath = argv[3];
    const std::string outputPath = (argc == 5) ? argv[4] : "./yolox_deepsort_output.mp4";
    const CliOptions options{};

    try
    {
        LOG.info("========================================");
        LOG.info("GryFlux YOLOX DeepSORT Pipeline");
        LOG.info("Yolox : %s", yoloxModelPath.c_str());
        LOG.info("ReID  : %s", reidModelPath.c_str());
        LOG.info("Input : %s", inputPath.c_str());
        LOG.info("Output: %s", outputPath.c_str());
        LOG.info("========================================");

        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();

        auto detectionContexts = CreateDetectionInferContexts(
            yoloxModelPath,
            options.deviceId,
            options.detectionNpuInstances);
        auto detectionProbeContext =
            std::static_pointer_cast<InferContext>(detectionContexts.front());
        const int detectionModelWidth = detectionProbeContext->getModelWidth();
        const int detectionModelHeight = detectionProbeContext->getModelHeight();
        resourcePool->registerResourceType("detector_npu", std::move(detectionContexts));

        auto reidContexts = CreateReidInferContexts(
            reidModelPath,
            options.deviceId,
            options.reidNpuInstances);
        auto reidProbeContext =
            std::static_pointer_cast<ReidContext>(reidContexts.front());
        const int reidModelWidth = reidProbeContext->getModelWidth();
        const int reidModelHeight = reidProbeContext->getModelHeight();
        resourcePool->registerResourceType("reid_npu", std::move(reidContexts));

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder *builder)
            {
                builder->setInputNode<PipelineNodes::InputNode>("input");
                builder->addTask<PipelineNodes::PreprocessNode>(
                    "preprocess",
                    "",
                    {"input"},
                    static_cast<std::size_t>(detectionModelWidth),
                    static_cast<std::size_t>(detectionModelHeight));
                builder->addTask<PipelineNodes::DetectionInferenceNode>(
                    "detection_inference",
                    "detector_npu",
                    {"preprocess"});
                builder->addTask<PipelineNodes::PostprocessNode>(
                    "postprocess",
                    "",
                    {"detection_inference"},
                    options.classCount,
                    options.confThreshold,
                    options.nmsThreshold,
                    options.maxDetections);
                builder->addTask<PipelineNodes::ReidPreprocessNode>(
                    "reid_preprocess",
                    "",
                    {"postprocess"},
                    reidModelWidth,
                    reidModelHeight);
                builder->addTask<PipelineNodes::ReidInferenceNode>(
                    "reid_inference",
                    "reid_npu",
                    {"reid_preprocess"},
                    options.reidFeatureDim);
                builder->setOutputNode<PipelineNodes::OutputNode>(
                    "output",
                    {"reid_inference"});
            });

        auto source = std::make_shared<VideoSource>(
            inputPath,
            options.maxDetections,
            reidModelWidth,
            reidModelHeight,
            options.reidFeatureDim);
        auto consumer = std::make_shared<ResultConsumer>(
            outputPath,
            source->getFps(),
            source->getWidth(),
            source->getHeight());

        GryFlux::AsyncPipeline pipeline(
            source,
            graphTemplate,
            resourcePool,
            consumer,
            options.threadPoolSize,
            options.maxActivePackets);

        if (options.enableProfiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.setProfilingEnabled(true);
            }
            else
            {
                LOG.info("Graph profiler not compiled, ignore --profile");
            }
        }

        const auto start = std::chrono::steady_clock::now();
        pipeline.run();
        const auto end = std::chrono::steady_clock::now();

        const auto costMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        const double seconds = static_cast<double>(costMs) / 1000.0;
        const std::size_t written = consumer->getWrittenCount();
        const double throughput = (seconds > 0.0) ? (static_cast<double>(written) / seconds) : 0.0;

        LOG.info("========================================");
        LOG.info("Pipeline done in %lld ms", static_cast<long long>(costMs));
        LOG.info("Written: %zu, throughput: %.2f packets/s", written, throughput);
        LOG.info("========================================");

        if (options.enableProfiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.printProfilingStats();
                const std::string timelinePath = "yolox_deepsort_graph_timeline.json";
                pipeline.dumpProfilingTimeline(timelinePath);
                LOG.info("Graph timeline dumped to %s", timelinePath.c_str());
            }
        }
    }
    catch (const std::exception &e)
    {
        LOG.error("Fatal error: %s", e.what());
        return -1;
    }

    return 0;
}
