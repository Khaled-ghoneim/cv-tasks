# Computer Vision Webcam Tasks

This project contains three OpenCV programs that read video from the computer's default webcam. Each program opens a preview window; press **q** while that window is focused to quit.

## Requirements

- Python 3.9 or newer
- A webcam connected to the computer, with camera access enabled for the terminal or IDE
- The packages listed in [`requirements.txt`](requirements.txt)

The YOLO demo also needs the two model weight files included in `task2/`: `yolov8n-face.pt` and `yolov8n-pose.pt`.

## Set up

Run these commands from the project root (the directory containing this README):

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run a task

Run one command from the project root:

```bash
# Show the original webcam image beside its grayscale conversion
python "task 1/task1.py"

# Detect faces and eyes with OpenCV Haar Cascades
python task2/task2.py

# Detect faces and estimate eye landmarks with YOLO models
python task2/yolo_face_eye.py
```

The scripts load cascade and model files from their own folders, so commands can be launched from the project root as shown above.

## What each script does

- `task 1/task1.py` displays a live camera frame beside its grayscale conversion. It skips failed frames and exits after more than five failed reads.
- `task2/task2.py` detects faces with OpenCV's frontal-face cascade, then searches for eyes inside each detected face.
- `task2/yolo_face_eye.py` draws YOLO face detections and eye landmarks estimated by the YOLO pose model. Eye landmarks are shown only when they fall inside a detected face.

## Troubleshooting

- **Camera cannot be opened:** close other apps using the webcam and allow camera access to your terminal or IDE. If the camera is not device `0`, change `VideoCapture(0)` in the relevant script.
- **YOLO weights cannot be loaded:** confirm both `.pt` files are in `task2/`.
- **A package is missing:** activate the virtual environment and install `requirements.txt` again.
