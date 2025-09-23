/**
 * utils.cpp
 * Author: Niloofar Karimi
 * Description: Implements utility functions used across the facial emotion recognition project.
 *              Includes preprocessing operations like BGR-to-grayscale conversion.
 */

#include "utils.hpp"
#include <opencv2/imgproc.hpp>

namespace Utils {

    // Convert a BGR image to grayscale using OpenCV
    cv::Mat toGrayscale(const cv::Mat& input) {
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

}
