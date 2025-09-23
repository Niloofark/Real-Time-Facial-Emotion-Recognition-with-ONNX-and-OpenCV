/**
 * face_detector.cpp
 * Author: Niloofar Karimi
 * Description: Implementation of the FaceDetector class.
 *              Uses OpenCV's Haar cascade to detect faces in grayscale images.
 */

#include "face_detector.hpp"
#include <stdexcept>  // For std::runtime_error

// Constructor: Loads the Haar cascade model from the given path.
// Throws an error if loading fails.
FaceDetector::FaceDetector(const std::string& cascadePath) {
    if (!faceCascade.load(cascadePath)) {
        throw std::runtime_error("Failed to load Haar cascade from path: " + cascadePath);
    }
}

// detect(): Detects faces in the provided grayscale frame
// Parameters:
//   - frameGray: input image in grayscale
// Returns:
//   - A vector of bounding rectangles for each detected face
std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& frameGray) {
    std::vector<cv::Rect> faces;

    // Run the Haar cascade classifier to detect faces
    // scaleFactor = 1.1: image is scaled down by 10% at each scale
    // minNeighbors = 3: a candidate rectangle needs 3 neighbors to be retained
    // flags = 0: use default flags
    // minSize = (30, 30): ignore faces smaller than 30x30 pixels
    faceCascade.detectMultiScale(frameGray, faces, 1.1, 3, 0, cv::Size(30, 30));

    return faces;
}
