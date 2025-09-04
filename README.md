# Organizing-frames-based-on-detection-confidence
A Python tool for extracting video frames and organizing them based on object detection confidence scores using a YOLO model
# YOLO Video Frame Organizer

This Python tool processes video files by extracting frames at a fixed interval and running object detection using a YOLO model (via [Ultralytics](https://github.com/ultralytics/ultralytics)). The extracted frames are automatically organized into folders based on the **maximum detection confidence score** in each frame.

## 🔍 Features

- Extracts frames from any video file at a specified FPS
- Resizes frames for faster processing
- Uses a YOLO model to detect objects in each frame
- Sorts frames into folders based on confidence score (e.g., `0.5-0.6`, `0.6-0.7`, etc.)
- Saves frames with high JPEG quality for later review

## 📂 Output Structure

