from ultralytics import YOLO
import cv2

model = YOLO("runs/classify/runs/fight_cls/weights/best.pt")
cap = cv2.VideoCapture("test_video.mp4")

correct = 0
total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    probs = results[0].probs
    if probs is not None:
        top1 = int(probs.top1)
        confidence = float(probs.top1conf)
        label = model.names[top1]

        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
        cv2.putText(frame, f"{label} {confidence:.0%}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        if label == "Violence":
            correct += 1
        total += 1

    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Fight detected in {correct}/{total} frames ({correct/total*100:.1f}%)")