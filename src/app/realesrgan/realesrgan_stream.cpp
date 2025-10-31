#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "framework/data_object.h"
#include "framework/processing_task.h"
#include "framework/streaming_pipeline.h"
#include "utils/logger.h"
#include "utils/unified_allocator.h"

#include "package.h"
#include "sink/write_consumer/write_consumer.h"
#include "source/producer/image_producer.h"
#include "tasks/image_preprocess/image_preprocess.h"
#include "tasks/res_sender/res_sender.h"
#include "tasks/rk_runner/rk_runner.h"

namespace SR = GryFlux::RealESRGAN;

namespace {

void initLogger() {
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::BOTH);
    LOG.setAppName("RealESRGANStream");

    const std::filesystem::path dirPath("./logs");
    if (!std::filesystem::exists(dirPath)) {
        try {
            std::filesystem::create_directories(dirPath);
        } catch (const std::exception &e) {
            LOG.error("[RealESRGANStream] Failed to create log directory: %s", e.what());
        }
    }
    LOG.setLogFileRoot("./logs");
}

void buildStreamingComputeGraph(std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                std::shared_ptr<GryFlux::DataObject> input,
                                const std::string &outputId,
                                GryFlux::TaskRegistry &taskRegistry) {
    auto inputNode = builder->addInput("input", input);
    auto preprocessNode = builder->addTask("imagePreprocess",
                                           taskRegistry.getProcessFunction("imagePreprocess"),
                                           {inputNode});
    auto runnerNode = builder->addTask("rkRunner",
                                       taskRegistry.getProcessFunction("rkRunner"),
                                       {preprocessNode});

    builder->addTask(outputId,
                     taskRegistry.getProcessFunction("resultSender"),
                     {inputNode, preprocessNode, runnerNode});
}

} // namespace

int main(int argc, const char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <dataset_path> [output_dir]" << std::endl;
        return 1;
    }

    initLogger();

    GryFlux::TaskRegistry taskRegistry;
    auto cpuAllocator = std::make_unique<CPUAllocator>();

    const std::size_t model_width = 256;
    const std::size_t model_height = 256;

    taskRegistry.registerTask<SR::ImagePreprocess>("imagePreprocess", static_cast<int>(model_width), static_cast<int>(model_height));
    taskRegistry.registerTask<SR::RkRunner>("rkRunner", argv[1], 1, model_width, model_height);
    taskRegistry.registerTask<SR::ResSender>("resultSender");

    GryFlux::StreamingPipeline pipeline(4);
    pipeline.setOutputNodeId("resultSender");
    pipeline.enableProfiling(true);

    pipeline.setProcessor([&taskRegistry](std::shared_ptr<GryFlux::PipelineBuilder> builder,
                                          std::shared_ptr<GryFlux::DataObject> input,
                                          const std::string &outputId) {
        buildStreamingComputeGraph(builder, input, outputId, taskRegistry);
    });

    pipeline.start();

    std::atomic<bool> running(true);
    const std::string dataset_path = argv[2];
    const std::string output_path = (argc == 4) ? argv[3] : "./outputs";

    SR::ImageProducer producer(pipeline, running, cpuAllocator.get(), dataset_path);
    SR::WriteConsumer consumer(pipeline, running, cpuAllocator.get(), output_path);

    producer.start();
    consumer.start();

    producer.join();
    LOG.info("[RealESRGANStream] Producer finished");

    running.store(false);

    consumer.join();
    LOG.info("[RealESRGANStream] Consumer finished, processed %d frames", consumer.getProcessedFrames());

    pipeline.stop();
    LOG.info("[RealESRGANStream] Pipeline stopped");

    return 0;
}
