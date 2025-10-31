#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "framework/data_object.h"
#include "framework/processing_task.h"
#include "framework/streaming_pipeline.h"

#include "utils/logger.h"
#include "utils/unified_allocator.h"

#include "package/package.h"
#include "sink/write_consumer/write_consumer.h"
#include "source/producer/fusion_image_producer.h"
#include "tasks/fusion_composer/fusion_composer.h"
#include "tasks/image_preprocess/image_preprocess.h"
#include "tasks/res_sender/res_sender.h"
#include "tasks/rk_runner/rk_runner.h"

namespace
{
    void buildStreamingComputeGraph(std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                    std::shared_ptr<GryFlux::DataObject> input,
                                    const std::string &outputId,
                                    GryFlux::TaskRegistry &taskRegistry)
    {
        auto inputNode = builder->addInput("input", input);
        auto preprocessNode = builder->addTask("imagePreprocess",
                                               taskRegistry.getProcessFunction("imagePreprocess"),
                                               {inputNode});
        auto rkNode = builder->addTask("rkRunner",
                                       taskRegistry.getProcessFunction("rkRunner"),
                                       {preprocessNode});
        auto composerNode = builder->addTask("fusionComposer",
                                             taskRegistry.getProcessFunction("fusionComposer"),
                                             {preprocessNode, rkNode});

        builder->addTask(outputId,
                         taskRegistry.getProcessFunction("resultSender"),
                         {composerNode});
    }

    void initLogger()
    {
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);

        std::filesystem::path dirPath("./logs");
        if (!std::filesystem::exists(dirPath))
        {
            try
            {
                std::filesystem::create_directories(dirPath);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to create log directory: " << e.what() << std::endl;
            }
        }
        try
        {
            auto absoluteRoot = std::filesystem::absolute(dirPath);
            LOG.setLogFileRoot(absoluteRoot.string());
            LOG.setAppName("FusionNetV2Stream");
            LOG.setOutputType(GryFlux::LogOutputType::BOTH);
        }
        catch (const std::exception &e)
        {
            LOG.warning("Failed to set log file root, fallback to console only: %s", e.what());
            LOG.setAppName("FusionNetV2Stream");
            LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
        }
    }
}

int main(int argc, const char **argv)
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <dataset_root> [output_dir]" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string datasetRoot = argv[2];
    std::string outputDir = (argc == 4) ? argv[3] : "./fusion_outputs";

    initLogger();

    GryFlux::TaskRegistry taskRegistry;

    auto allocator = std::make_unique<CPUAllocator>();

    constexpr int kModelWidth = 640;
    constexpr int kModelHeight = 480;

    try
    {
        taskRegistry.registerTask<GryFlux::ImagePreprocess>("imagePreprocess", kModelWidth, kModelHeight);
        taskRegistry.registerTask<GryFlux::RkRunner>("rkRunner", modelPath);
        taskRegistry.registerTask<GryFlux::FusionComposer>("fusionComposer");
        taskRegistry.registerTask<GryFlux::ResSender>("resultSender");
    }
    catch (const std::exception &e)
    {
        LOG.error("Failed to initialize tasks: %s", e.what());
        std::cerr << "Failed to initialize pipeline tasks: " << e.what() << std::endl;
        return 1;
    }

    GryFlux::StreamingPipeline pipeline(8);
    pipeline.setOutputNodeId("resultSender");
    pipeline.enableProfiling(true);

    pipeline.setProcessor([&taskRegistry](std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                          std::shared_ptr<GryFlux::DataObject> input,
                                          const std::string &outputId)
                          { buildStreamingComputeGraph(builder, input, outputId, taskRegistry); });

    pipeline.start();

    std::atomic<bool> running(true);

    GryFlux::FusionImageProducer producer(pipeline, running, allocator.get(), datasetRoot);
    GryFlux::WriteConsumer consumer(pipeline, running, allocator.get(), outputDir);

    if (!producer.start())
    {
        LOG.error("Failed to start producer thread");
        pipeline.stop();
        return 1;
    }
    if (!consumer.start())
    {
        LOG.error("Failed to start consumer thread");
        pipeline.stop();
        producer.stop();
        producer.join();
        return 1;
    }

    producer.join();
    LOG.info("[main] Producer finished");

    consumer.join();
    LOG.info("[main] Consumer finished, processed %d frames", consumer.getProcessedFrames());

    pipeline.stop();
    LOG.info("[main] Pipeline stopped");

    return 0;
}
