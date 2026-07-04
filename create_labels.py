import os
from pathlib import Path
import shutil
import random

# Settings
PREPARED_DIR = "dataset_prepared/images"
OUTPUT_DIR = "dataset_yolo"
SPLIT = 0.8  # 80% training, 20% validation

classes = ["NonViolence", "Violence"]

# Create output folders
for split in ["train", "val"]:
    for folder in ["images", "labels"]:
        Path(f"{OUTPUT_DIR}/{split}/{folder}").mkdir(parents=True, exist_ok=True)

for class_idx, class_name in enumerate(classes):
    images = list(Path(f"{PREPARED_DIR}/{class_name}").glob("*.jpg"))
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img_list, split in [(train_images, "train"), (val_images, "val")]:
        for img_path in img_list:
            # Copy image
            shutil.copy(img_path, f"{OUTPUT_DIR}/{split}/images/{img_path.name}")

            # Create label file (whole image as bounding box)
            label_path = f"{OUTPUT_DIR}/{split}/labels/{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

    print(f"✅ Done {class_name} — Train: {len(train_images)}, Val: {len(val_images)}")

# Create dataset.yaml
with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
    f.write(f"""path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images

nc: 2
names: ['NonViolence', 'Violence']
""")

print("✅ Dataset ready for YOLOv8 training!")