# ------------------------------------------------------------
# YOLO Video Frame Organizer
# Description:
#   Extracts frames from a video, detects objects using a YOLO model,
#   and saves frames into folders based on detection confidence scores.
# ------------------------------------------------------------

import os
import cv2
import shutil
from ultralytics import YOLO  # Make sure ultralytics is installed (pip install ultralytics)

def get_confidence_folder(conf):
    """
    Returns a folder name string based on the confidence score.
    E.g., 0.65 → '0.6-0.7'
    """
    lower = round(int(conf * 10) / 10, 1)
    upper = round(lower + 0.1, 1)
    return f"{lower:.1f}-{upper:.1f}"


def extractFramesFromVideo(videoPath, outputBaseFolder, modelName,
                           target_fps=3, frame_dimensions=(640, 380), image_quality=99):
    """
    Extracts frames from a video and organizes them into folders based on detection confidence.

    Parameters:
        videoPath (str): Path to the input video file.
        outputBaseFolder (str): Base folder to save extracted frames.
        modelName (str): Path to YOLO model (.pt file).
        target_fps (int): Number of frames to extract per second.
        frame_dimensions (tuple): Size to resize frames (width, height).
        image_quality (int): JPEG image quality (1-100).
    """
    try:
        # Load YOLO model
        model = YOLO(modelName)

        # Open the video file
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {videoPath}")
            return

        # Get original video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / target_fps) if fps > 0 else 1

        frame_count = 0
        saved_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Process frame only at specified intervals
            if frame_count % frame_interval == 0:
                # Resize frame
                resized_frame = cv2.resize(frame, frame_dimensions)

                # Run YOLO object detection
                results = model.predict(source=resized_frame, verbose=False)
                boxes = results[0].boxes

                # Determine target folder based on confidence
                if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
                    targetFolder = os.path.join(outputBaseFolder, "no_detections")
                else:
                    confidences = [conf.item() for conf in boxes.conf]
                    max_conf = max(confidences)
                    print(f"Frame {saved_frame_count} — max_conf: {round(max_conf, 4)}")
                    folder_name = get_confidence_folder(max_conf)
                    targetFolder = os.path.join(outputBaseFolder, folder_name)

                # Create target folder if it doesn't exist
                os.makedirs(targetFolder, exist_ok=True)

                # Save frame as JPEG
                frame_filename = f"frame_{saved_frame_count:05d}.jpg"
                frame_path = os.path.join(targetFolder, frame_filename)
                cv2.imwrite(frame_path, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), image_quality])
                print(f"Saved: {frame_filename} → {targetFolder}")

                saved_frame_count += 1

            frame_count += 1

        cap.release()
        print("Finished processing video.")

    except Exception as e:
        print(f"An error occurred: {e}")


# Entry point of the script
if __name__ == "__main__":
    # Set your paths here
    videoPath = ''
    outputBaseFolder = 'outputFolder'
    modelName = 'best.pt'

    # Ensure output directory exists
    os.makedirs(outputBaseFolder, exist_ok=True)

    # Start processing
    extractFramesFromVideo(videoPath, outputBaseFolder, modelName)
