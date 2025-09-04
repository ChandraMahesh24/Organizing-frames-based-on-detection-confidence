import os
import cv2
import shutil
from ultralytics import YOLO

def get_confidence_folder(conf):
    lower = round(int(conf * 10) / 10, 1)
    upper = round(lower + 0.1, 1)
    return f"{lower:.1f}-{upper:.1f}"

def extractFramesFromVideo(videoPath, outputBaseFolder, modelName, target_fps=3, frame_dimensions=(640, 380), image_quality=99):
    try:
        model = YOLO(modelName)

        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {videoPath}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / target_fps) if fps > 0 else 1

        frame_count = 0
        saved_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                resized_frame = cv2.resize(frame, frame_dimensions)
                results = model.predict(source=resized_frame, verbose=False)
                boxes = results[0].boxes

                if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
                    targetFolder = os.path.join(outputBaseFolder, "no_detections")
                else:
                    confidences = [conf.item() for conf in boxes.conf]
                    max_conf = max(confidences)
                    print(f"Frame {saved_frame_count} — max_conf: {round(max_conf, 4)}")
                    folder_name = get_confidence_folder(max_conf)
                    targetFolder = os.path.join(outputBaseFolder, folder_name)

                os.makedirs(targetFolder, exist_ok=True)

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

if __name__ == "__main__":
    videoPath = r'e:\kmg\kmg_ch1_20250902083024_20250902085207.mp4'
    outputBaseFolder = r'e:\kmg\output\KGM 2\organized_by_conf_3'
    modelName = r'c:\Users\Admin\Downloads\best (8).pt'

    os.makedirs(outputBaseFolder, exist_ok=True)

    extractFramesFromVideo(videoPath, outputBaseFolder, modelName)


