import cv2
import os
from pathlib import Path

# Settings
DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset_prepared"
FRAMES_PER_VIDEO = 10  # Extract 10 frames per video

classes = ["NonViolence", "Violence"]

for class_name in classes:
    class_idx = classes.index(class_name)
    video_dir = Path(DATASET_DIR) / class_name
    output_img_dir = Path(OUTPUT_DIR) / "images" / class_name
    output_img_dir.mkdir(parents=True, exist_ok=True)

    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    print(f"Processing {len(videos)} videos for {class_name}...")

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // FRAMES_PER_VIDEO)

        frame_count = 0
        saved = 0

        while saved < FRAMES_PER_VIDEO:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            filename = f"{video_path.stem}_frame{saved}.jpg"
            cv2.imwrite(str(output_img_dir / filename), frame)
            frame_count += step
            saved += 1

        cap.release()

    print(f"✅ Done {class_name}")

print("✅ All frames extracted!") 