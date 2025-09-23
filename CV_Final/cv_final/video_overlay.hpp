/**
 * video_overlay.hpp
 * Author: Niloofar Karimi
 * Description: Header file for drawing overlays on the video stream.
 *              Provides functionality to annotate detected faces with emotion labels and confidence.
 */

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace VideoOverlay {
    // Draws rectangles and labels (with confidence) on detected faces in the frame
    void drawDetections(cv::Mat& frame,
                        const std::vector<cv::Rect>& faces,
                        const std::vector<std::string>& labels,
                        const std::vector<float>& confidences);
}
