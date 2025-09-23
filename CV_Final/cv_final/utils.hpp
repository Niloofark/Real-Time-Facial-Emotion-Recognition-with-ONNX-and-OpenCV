/**
 * utils.hpp
 * Author: Niloofar Karimi
 * Description: Header file for utility functions used in facial emotion recognition,
 *              such as image preprocessing helpers.
 */

#pragma once
#include <opencv2/core.hpp>

// Utility functions for image processing
namespace Utils {
    // Convert a BGR image to grayscale (used before face detection/classification)
    cv::Mat toGrayscale(const cv::Mat& input);
}
