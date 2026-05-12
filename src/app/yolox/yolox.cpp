#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/image_consumer.h"
#include "app/common/rknn_context.h"
#include "nodes/yolox_nodes.h"
#include "source/image_dir_source.h"

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
    float confThreshold = 0.5f;
    float nmsThreshold = 0.6f;
    std::size_t maxDetections = 80;
    std::size_t npuInstances = 3;
    std::size_t threadPoolSize = 8;
    std::size_t maxActivePackets = 8;
    bool enableProfiling = true;
};

void printHelp()
{
    LOG.info("Usage: yolox <model_path> <dataset_dir> [output_dir]");
    LOG.info("Pipeline config is defined in CliOptions in src/app/yolox/yolox.cpp");
}
} // namespace

int main(int argc, char **argv)
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("yolox");

    if (argc < 3 || argc > 4)
    {
        printHelp();
        return -1;
    }

    const std::string modelPath = argv[1];
    const std::string datasetDir = argv[2];
    const std::string outputDir = (argc == 4) ? argv[3] : "./outputs";
    const CliOptions options{};

    try
    {
        LOG.info("========================================");
        LOG.info("GryFlux Yolox Pipeline");
        LOG.info("Model : %s", modelPath.c_str());
        LOG.info("Input : %s", datasetDir.c_str());
        LOG.info("Output: %s", outputDir.c_str());
        LOG.info("========================================");

        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        auto probeContext = std::make_shared<RKNNContext>(0, modelPath);
        const int modelWidth = probeContext->getModelWidth();
        const int modelHeight = probeContext->getModelHeight();

        {
            std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
            npuContexts.reserve(options.npuInstances);
            npuContexts.push_back(probeContext);
            for (std::size_t i = 1; i < options.npuInstances; ++i)
            {
                npuContexts.push_back(std::make_shared<RKNNContext>(
                    static_cast<int>(i),
                    modelPath,
                    modelWidth,
                    modelHeight));
            }
            resourcePool->registerResourceType("npu", std::move(npuContexts));
        }

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder *builder)
            {
                builder->setInputNode<YoloxNodes::InputNode>("input");
                builder->addTask<YoloxNodes::PreprocessNode>(
                    "preprocess",
                    "",
                    {"input"},
                    static_cast<std::size_t>(modelWidth),
                    static_cast<std::size_t>(modelHeight));
                builder->addTask<YoloxNodes::InferenceNode>("inference", "npu", {"preprocess"});
                builder->addTask<YoloxNodes::PostprocessNode>(
                    "postprocess",
                    "",
                    {"inference"},
                    options.classCount,
                    options.confThreshold,
                    options.nmsThreshold,
                    options.maxDetections);
                builder->setOutputNode<YoloxNodes::OutputNode>("output", {"postprocess"});
            });

        auto source = std::make_shared<YoloxImageDirSource>(datasetDir);
        auto consumer = std::make_shared<YoloxImageConsumer>(outputDir);

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
                const std::string timelinePath = "yolox_graph_timeline.json";
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
