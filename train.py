from ultralytics import YOLO

if __name__ == '__main__':
    # Use classification model instead of detection
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="dataset_prepared/images",
        epochs=20,
        imgsz=416,
        batch=32,
        device=0,
        project="runs",
        name="fight_cls",
        patience=5,
        workers=0
    )

    print("✅ Training complete!")