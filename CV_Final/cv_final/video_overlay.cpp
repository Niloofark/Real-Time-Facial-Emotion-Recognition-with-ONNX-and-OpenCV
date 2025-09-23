/**
 * video_overlay.cpp
 * Author: Niloofar Karimi
 * Description: Implements functionality for drawing overlays on the video stream.
 *              This includes drawing bounding boxes around detected faces and annotating
 *              them with predicted emotion labels and confidence scores.
 */

#include "video_overlay.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>
#include <sstream>

namespace VideoOverlay {

    // Draw bounding boxes and emotion labels (with confidence) on the frame
    void drawDetections(cv::Mat& frame,
                        const std::vector<cv::Rect>& faces,
                        const std::vector<std::string>& labels,
                        const std::vector<float>& confidences) {
        for (size_t i = 0; i < faces.size(); ++i) {
            const cv::Rect& rect = faces[i];
            const std::string& label = labels[i];

            // Format label with confidence as percentage (if available)
            std::ostringstream oss;
            if (i < confidences.size()) {
                oss << label << " (" << std::fixed << std::setprecision(2) << confidences[i] * 100 << "%)";
            } else {
                oss << label;
            }
            std::string labelText = oss.str();

            // Draw bounding box around the face in green
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);

            // Calculate size of the label text
            int baseline = 0;
            cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            // Draw green background rectangle behind label
            cv::Rect background(rect.x, rect.y - labelSize.height - 5, labelSize.width + 4, labelSize.height + 4);
            cv::rectangle(frame, background, cv::Scalar(0, 255, 0), cv::FILLED);

            // Draw label text in black
            cv::putText(frame, labelText, cv::Point(rect.x + 2, rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }

}
