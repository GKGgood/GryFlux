#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/realesrgan_packet.h"

#include <opencv2/opencv.hpp>

#include <stdexcept>

namespace RealesrganNodes
{

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealesrganPacket &>(packet);

    if (p.srTensorF32.empty())
    {
        throw std::runtime_error("Postprocess: empty SR tensor");
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(p.srTensorF32, &minVal, &maxVal);

    cv::Mat scaled;
    const double scale = (maxVal <= 2.0) ? 255.0 : 1.0;
    p.srTensorF32.convertTo(scaled, CV_32FC3, scale);
    cv::max(scaled, 0.0, scaled);
    cv::min(scaled, 255.0, scaled);

    cv::Mat srU8;
    scaled.convertTo(srU8, CV_8UC3);

    cv::cvtColor(srU8, p.outputBgrU8, cv::COLOR_RGB2BGR);
}

} // namespace RealesrganNodes
