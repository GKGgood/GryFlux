#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "app/common/rknn_context.h"
#include "consumer/result_consumer.h"
#include "nodes/resnet_nodes.h"
#include "source/image_dir_source.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct CliOptions
{
    std::size_t topK = 5;
    std::size_t npuInstances = 3;
    std::size_t threadPoolSize = 8;
    std::size_t maxActivePackets = 8;
    bool enableProfiling = true;
};

void printHelp()
{
    LOG.info("Usage: resnet <model_path> <dataset_dir> <synset_path> [output_dir]");
    LOG.info("Pipeline config is defined in CliOptions in src/app/resnet/resnet.cpp");
}

std::vector<std::string> loadLabels(const std::string &synsetPath)
{
    std::ifstream file(synsetPath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open synset file: " + synsetPath);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line))
    {
        const std::size_t firstSpace = line.find(' ');
        if (firstSpace != std::string::npos && firstSpace + 1 < line.size())
        {
            labels.push_back(line.substr(firstSpace + 1));
        }
        else
        {
            labels.push_back(line);
        }
    }
    return labels;
}
} // namespace

int main(int argc, char **argv)
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("resnet");

    if (argc < 4 || argc > 5)
    {
        printHelp();
        return -1;
    }

    try
    {
        const std::string modelPath = argv[1];
        const std::string datasetDir = argv[2];
        const std::string synsetPath = argv[3];
        const std::string outputDir = (argc == 5) ? argv[4] : "./outputs";
        const CliOptions options{};

        const auto classLabels = loadLabels(synsetPath);
        if (classLabels.empty())
        {
            LOG.warning("Synset file is empty, fallback labels will use class_<id>");
        }

        LOG.info("========================================");
        LOG.info("GryFlux ResNet Pipeline");
        LOG.info("Model : %s", modelPath.c_str());
        LOG.info("Input : %s", datasetDir.c_str());
        LOG.info("Synset: %s", synsetPath.c_str());
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
                builder->setInputNode<ResnetNodes::InputNode>("input");
                builder->addTask<ResnetNodes::PreprocessNode>("preprocess", "", {"input"}, modelWidth, modelHeight);
                builder->addTask<ResnetNodes::InferenceNode>("inference", "npu", {"preprocess"});
                builder->addTask<ResnetNodes::PostprocessNode>(
                    "postprocess",
                    "",
                    {"inference"},
                    classLabels,
                    options.topK);
                builder->setOutputNode<ResnetNodes::OutputNode>("output", {"postprocess"});
            });

        auto source = std::make_shared<ResnetImageDirSource>(datasetDir);
        auto consumer = std::make_shared<ResnetResultConsumer>(outputDir);

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
        const std::size_t consumed = consumer->getConsumedCount();
        const std::size_t written = consumer->getWrittenCount();
        const double throughput = (seconds > 0.0) ? (static_cast<double>(consumed) / seconds) : 0.0;

        LOG.info("========================================");
        LOG.info("Pipeline done in %lld ms", static_cast<long long>(costMs));
        LOG.info("Consumed: %zu, written: %zu, throughput: %.2f packets/s",
                 consumed,
                 written,
                 throughput);
        LOG.info("========================================");

        if (options.enableProfiling)
        {
            if constexpr (GryFlux::Profiling::kBuildProfiling)
            {
                pipeline.printProfilingStats();
                const std::string timelinePath = "resnet_graph_timeline.json";
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
