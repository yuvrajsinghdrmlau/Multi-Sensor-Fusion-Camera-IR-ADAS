# Multi-Sensor-Fusion-Camera-IR-ADAS
# Multi-Sensor Fusion for Object Detection (Camera + IR)

## Overview
This project implements a multi-sensor fusion pipeline combining RGB camera and infrared (IR) data to improve object detection robustness in low-light and night-time ADAS scenarios.

## Motivation
Single-sensor perception systems struggle under poor illumination. By fusing RGB and IR modalities, the system achieves higher robustness and reliability for automotive perception tasks.

## Methodology
- Sensor calibration and spatial alignment between RGB and IR inputs
- Dual-backbone architecture with modality-specific feature extraction
- Feature-level fusion using attention mechanisms
- Temporal consistency across video frames to reduce false positives

## Results
- ~15–20% improvement in detection accuracy compared to single-sensor models
- Stable detections under low illumination and night-time conditions
- Real-time inference performance of ~25–30 FPS on GPU

## Tech Stack
- Python, PyTorch
- OpenCV
- CUDA (GPU acceleration)

## Applications
- ADAS perception
- Night-time pedestrian and obstacle detection
- Autonomous driving sensor fusion

## Future Work
- Extension to radar-camera fusion
- ROS2 integration
- Embedded deployment (TensorRT)
