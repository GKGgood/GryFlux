#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "app/common/rknn_context.h"
#include "consumer/ZeroDceResultConsumer.h"
#include "nodes/Infer/InferNode.h"
#include "nodes/Postprocess/PostprocessNode.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "source/ZeroDceDataSource.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct AppConfig
{
    std::string model_path;
    std::string input_dir;
    std::string output_dir;
    std::size_t npu_instances = 3;
    std::size_t thread_pool_size = 8;
    std::size_t max_active_packets = 16;
    bool enable_profiling = true;
};

void printUsage(const char *programName)
{
    LOG.info("Usage: %s <model_path> <input_dir> <output_dir>", programName);
    LOG.info("Pipeline config is defined in AppConfig in src/app/ZeroDCE/zero_dce.cpp");
}

bool parseArgs(int argc, char *argv[], AppConfig *config)
{
    if (argc != 4)
    {
        return false;
    }

    config->model_path = argv[1];
    config->input_dir = argv[2];
    config->output_dir = argv[3];
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("zero_dce");

    AppConfig config;
    if (!parseArgs(argc, argv, &config))
    {
        printUsage(argv[0]);
        return -1;
    }

    try
    {
        LOG.info("========================================");
        LOG.info("GryFlux ZeroDCE Pipeline");
        LOG.info("Model : %s", config.model_path.c_str());
        LOG.info("Input : %s", config.input_dir.c_str());
        LOG.info("Output: %s", config.output_dir.c_str());
        LOG.info("========================================");

        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        auto probeContext = std::make_shared<RKNNContext>(0, config.model_path);
        const int modelWidth = probeContext->getModelWidth();
        const int modelHeight = probeContext->getModelHeight();

        {
            std::vector<std::shared_ptr<GryFlux::Context>> npuContexts;
            npuContexts.reserve(config.npu_instances);
            npuContexts.push_back(probeContext);
            for (std::size_t i = 1; i < config.npu_instances; ++i)
            {
                npuContexts.push_back(std::make_shared<RKNNContext>(
                    static_cast<int>(i),
                    config.model_path,
                    modelWidth,
                    modelHeight));
            }
            resourcePool->registerResourceType("npu", std::move(npuContexts));
        }

        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder *builder)
            {
                builder->setInputNode<PreprocessNode>(
                    "preprocess",
                    static_cast<std::size_t>(modelWidth),
                    static_cast<std::size_t>(modelHeight));
                builder->addTask<InferNode>("inference", "npu", {"preprocess"});
                builder->setOutputNode<PostprocessNode>("postprocess", {"inference"});
            });

        auto source = std::make_shared<ZeroDceDataSource>(
            config.input_dir,
            modelWidth,
            modelHeight);
        auto consumer = std::make_shared<ZeroDceResultConsumer>(
            config.output_dir,
            source->GetTotalFrames());

        GryFlux::AsyncPipeline pipeline(
            source,
            graphTemplate,
            resourcePool,
            consumer,
            config.thread_pool_size,
            config.max_active_packets);

        if (config.enable_profiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.setProfilingEnabled(true);
            }
        }

        const auto startTime = std::chrono::steady_clock::now();
        pipeline.run();
        const auto endTime = std::chrono::steady_clock::now();

        const auto costMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        const double seconds = static_cast<double>(costMs) / 1000.0;
        const std::size_t written = consumer->getWrittenCount();
        const double throughput = (seconds > 0.0) ? (static_cast<double>(written) / seconds) : 0.0;
        std::cout << '\n';

        LOG.info("========================================");
        LOG.info("Pipeline done in %lld ms", static_cast<long long>(costMs));
        LOG.info("Written: %zu, throughput: %.2f packets/s", written, throughput);
        LOG.info("========================================");

        if (config.enable_profiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.printProfilingStats();
                const std::string timelinePath = "zero_dce_graph_timeline.json";
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
