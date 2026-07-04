# Gemmiz CCTV AI Monitor

Real-time AI-powered fight detection system using YOLOv8 and Flask.

## Features
- Real-time fight detection via webcam or CCTV
- Live dashboard with event log and snapshots
- AI status display with confidence score
- Automatic snapshot saving on fight detection
- SQLite event logging with timestamps

## Tech Stack
- **AI Model:** YOLOv8n-cls (92.6% accuracy)
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Database:** SQLite
- **Video Processing:** OpenCV

## Setup

### 1. Clone the repo
git clone https://github.com/Sambramshetty/cctv-ai-monitor.git
cd cctv-ai-monitor

### 2. Install dependencies
pip install ultralytics opencv-python flask

### 3. Download dataset
https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

Place videos in:
dataset/
├── Violence/
└── NonViolence/

### 4. Prepare and train
python prepare_dataset.py
python create_labels.py
python train.py

### 5. Run the app
python app.py

Open browser at: http://127.0.0.1:5000

## Model
- Architecture: YOLOv8n-cls
- Accuracy: 92.6% mAP50
- Classes: Violence / NonViolence
- Dataset: Real Life Violence Situations (Kaggle, 2000 videos)

## Powered by Gemmiz AI
