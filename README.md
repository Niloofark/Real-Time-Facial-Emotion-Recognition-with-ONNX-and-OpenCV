# Real-Time Facial Emotion Recognition with ONNX and OpenCV

**Author:** Niloofar Karimi  
**Email:** karimi.ni@northeastern.edu  
**Institution:** Northeastern University  
**Course:** Computer Vision Final Project – Spring 2025



## Project Description

This project implements a real-time facial emotion recognition system using a lightweight convolutional neural network (CNN) model exported to ONNX and deployed with OpenCV in C++. The system classifies live webcam input into one of seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

To enhance real-world performance and prediction reliability, the system integrates:

- **Face alignment** using Haar cascade eye detection  
- **Test-Time Augmentation (TTA)** via flipping and rotation  
- **Confidence filtering**: predictions with confidence below 0.2 are labeled “Uncertain”  
- **Temporal smoothing** with a majority vote buffer  
- **CSV logging** of frame-wise predictions for analysis


## Presentation & Demo

- **Presentation Video**: https://drive.google.com/drive/u/0/folders/1hBH7lQVuWiKAvAubBI_cEfOlCUd6kZWx
- **Sample Output CSV**: `results.csv` (auto-generated during live testing)  





## Build & Run Instructions

### Requirements

- C++17 compatible compiler (e.g., `g++`, Clang, or Xcode)
- OpenCV 4.x installed and properly linked
- ONNX model file `mini_xception.onnx` placed in `/models/`

### Build (Linux/macOS example with g++)

```bash
g++ -std=c++17 -o emotion_app \
    main.cpp emotion_classifier.cpp face_detector.cpp video_overlay.cpp utils.cpp \
    `pkg-config --cflags --libs opencv4`
```
### Run Main.cpp
- Press T to toggle Test-Time Augmentation (TTA) on/off
- Press ESC to exit
- Frame-by-frame predictions and confidence scores are saved in results.csv

cv_final/
│
├── main.cpp
│   ├─ Entry point for the real-time emotion recognition application  
│   ├─ Captures webcam input  
│   ├─ Performs face detection and emotion classification (with optional TTA)  
│   └─ Draws real-time overlays and logs results to CSV
│
├── batch_test.cpp
│   └─ (Optional) Script to evaluate emotion recognition on static images
│
├── config.hpp
│   └─ Contains global paths, constants, and emotion label definitions
│
├── emotion_classifier.hpp / emotion_classifier.cpp
│   ├─ Defines and implements the EmotionClassifier class  
│   ├─ Handles ONNX model loading, preprocessing, and inference  
│   └─ Includes Test-Time Augmentation (TTA) logic
│
├── face_detector.hpp / face_detector.cpp
│   ├─ Defines and implements the FaceDetector class using OpenCV Haar cascades  
│   └─ Responsible for real-time face and eye detection (used for alignment)
│
├── video_overlay.hpp / video_overlay.cpp
│   ├─ Provides overlay utilities for bounding boxes, labels, and confidence scores  
│   └─ Draws detection results on each video frame in real time
│
├── utils.hpp / utils.cpp
│   ├─ General-purpose utility functions  
│   └─ Includes image preprocessing (e.g., grayscale conversion, normalization)
│
├── models/
│   └─ mini_xception.onnx — Pretrained Mini-Xception model in ONNX format
│
├── resources/
│   ├─ haarcascade_frontalface_default.xml — OpenCV frontal face detector  
│   ├─ haarcascade_eye.xml — OpenCV eye detector for face alignment  
│   ├─ test_images/ — (Optional) Sample grayscale face images for offline testing  
│   └─ results.csv — Auto-generated CSV containing per-frame:
│        - Frame number  
│        - Predicted emotion label  
│        - Confidence score  
│        - TTA mode (Yes/No)
│
├── README.md — Project documentation (this file)
└── Niloofar_Karimi_CV_Final_Project_Report.pdf — Final project report



###  Evaluation Summary

The system was tested under multiple configurations, measuring accuracy improvements from different preprocessing and enhancement techniques:

| Configuration                             | Accuracy (%) |
|------------------------------------------|--------------|
| Baseline (Grayscale + Resize)            | 18.7         |
| + Normalization + Center Crop            | 25.2         |
| + Face Alignment                         | 50.4         |
| + TTA + Smoothing + Confidence Filtering | 61.6         |

Visual analysis of confidence scores showed improved consistency and stability with TTA enabled.




## Acknowledgments

This project was completed as part of the **Computer Vision Final Project** at Northeastern University (Spring 2025). Special thanks to the course instructors and peers for their support and feedback throughout development.
