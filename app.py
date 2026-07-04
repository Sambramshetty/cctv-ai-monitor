from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from database import init_db, log_event, get_events

app = Flask(__name__)

init_db()
os.makedirs("static/snapshots", exist_ok=True)

model = YOLO("runs/classify/runs/fight_cls/weights/best.pt")

fight_status = {"fight_detected": False, "label": "Normal", "confidence": 0}
current_frame = None

def generate_frames():
    global current_frame
    cap = cv2.VideoCapture(0)
    fight_cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        fight_detected = False
        label = "Normal"
        confidence = 0

        probs = results[0].probs
        if probs is not None:
            top1 = int(probs.top1)
            confidence = float(probs.top1conf)
            label = model.names[top1]
            if label == "Violence" and confidence > 0.85:
                fight_detected = True
                cv2.putText(frame, f"FIGHT! {confidence:.0%}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        fight_status["fight_detected"] = fight_detected
        fight_status["label"] = label
        fight_status["confidence"] = round(confidence * 100)
        current_frame = frame.copy()

        if fight_detected and fight_cooldown == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = f"static/snapshots/fight_{timestamp}.jpg"
            cv2.imwrite(snapshot_path, frame)
            log_event("Fight Detected", snapshot_path)
            fight_cooldown = 30

        if fight_cooldown > 0:
            fight_cooldown -= 1

        if fight_detected:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_status')
def detection_status():
    return jsonify(fight_status)

@app.route('/events')
def events():
    return jsonify(get_events())

@app.route('/snapshots')
def snapshots():
    snaps = []
    snap_dir = "static/snapshots"
    if os.path.exists(snap_dir):
        files = sorted(os.listdir(snap_dir), reverse=True)[:6]
        snaps = [f"/static/snapshots/{f}" for f in files]
    return jsonify(snaps)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True)