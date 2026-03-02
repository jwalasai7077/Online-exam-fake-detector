# Online Exam Fake Detector

An AI-powered online proctoring project that monitors a candidate through webcam/audio signals and flags suspicious behavior in real time.

This repository combines multiple computer-vision and audio-analysis experiments, plus an integrated proctoring pipeline, to detect events such as:
- Multiple faces in frame
- Eye-gaze deviation
- Head pose changes
- Mouth opening (possible speaking)
- Face spoofing attempts
- Presence of extra person/mobile phone (YOLO-based module)

---

## Project Overview

The core integrated script (`Proctor.py`) loads face detection, landmark estimation, and spoofing models, opens webcam/video input, and overlays alerts frame-by-frame.

There is also a FastAPI wrapper (`main.py`) that can start/stop the proctoring process through HTTP endpoints.

Besides the integrated pipeline, the repository contains standalone scripts for face detection comparison, eye tracking, head pose estimation, mouth opening detection, face spoofing, person/phone detection, and audio-based speech overlap analysis.

---

## Key Capabilities

### Real-time integrated monitoring
`Proctor.py` provides a single-screen pipeline with:
- Face detection
- Eye tracking status (left/right/up)
- Head direction estimate (up/down)
- Mouth open/closed status with calibration
- Face spoofing probability
- Console warnings for suspicious events

### API-driven control
`main.py` exposes:
- `POST /analyze_video` to start analysis
- `POST /stop` to terminate running analysis process

### Modular experimentation
Independent scripts allow testing or improving each subsystem separately before integrating.

---

## Repository Structure

```text
.
├── Proctor.py                  # Integrated real-time proctoring pipeline
├── main.py                     # FastAPI service to start/stop proctoring
├── face_detector.py            # DNN-based face detection helpers
├── face_landmarks.py           # Facial landmark loading and inference helpers
├── eye_tracker.py              # Standalone eye-gaze tracking demo
├── head_pose_estimation.py     # Standalone head-pose estimation demo
├── mouth_opening_detector.py   # Standalone mouth-opening detection demo
├── face_spoofing.py            # Standalone spoof detection demo
├── person_and_phone.py         # YOLO-based person/mobile phone detection module
├── faces_detection.py          # Comparison script for multiple face detectors
├── audio_part.py               # Audio recording + speech-to-text overlap script
├── requirements.txt            # Python dependencies
├── test.txt                    # Intermediate audio text output
├── record0.wav/record1.wav/... # Sample recorded audio artifacts
└── eye_tracking/               # Sample eye tracking resources
```

---

## Tech Stack

- Python
- OpenCV (`opencv-python`)
- TensorFlow/Keras (landmark and YOLO-related components)
- scikit-learn + joblib (spoof classifier loading)
- FastAPI + Uvicorn (API wrapper)
- PyAudio + SpeechRecognition + NLTK (audio module)

---

## Prerequisites

- Python 3.9+ (recommended)
- Webcam (for live analysis)
- Optional microphone (for audio module)
- OS-level libraries required by OpenCV / PyAudio
- Model files placed under a `models/` directory (see below)

---

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Online-exam-fake-detector
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> Note: If dependency installation fails due to formatting/encoding issues in `requirements.txt`, create a clean UTF-8 copy and reinstall.

---

## Required Model Files

The code expects several files under `models/` that are **not included** in this repository snapshot.

Create a `models/` directory and place at least:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`
- `pose_model/` (TensorFlow SavedModel for facial landmarks)
- `face_spoofing.pkl`
- Any YOLO weights/config/classes required by `person_and_phone.py`

If these files are missing, scripts may fail during startup.

---

## How to Run

### Run the integrated proctoring app

```bash
python Proctor.py
```

- Press `c` to calibrate mouth baseline (keep mouth closed during calibration).
- Press `q` to quit.

You can also pass a video source argument:
```bash
python Proctor.py 0              # default webcam
python Proctor.py path/to/video.mp4
```

### Run as API

Start server:
```bash
uvicorn main:app --reload
```

Then call:
```bash
curl -X POST "http://127.0.0.1:8000/analyze_video?video_url=0"
curl -X POST "http://127.0.0.1:8000/stop"
```

> Important: `main.py` currently references `proctor.py` (lowercase), while this repository file is `Proctor.py` (uppercase). On case-sensitive systems (Linux), either rename the file or update `main.py`.

### Run individual experiment scripts

```bash
python eye_tracker.py
python head_pose_estimation.py
python mouth_opening_detector.py
python face_spoofing.py
python person_and_phone.py
python faces_detection.py
python audio_part.py
```

Each script may require additional assets (weights, dataset folders, text files, audio dependencies).

---

## API Endpoints

### `POST /analyze_video`
Starts proctoring in a subprocess.

**Query parameter**
- `video_url` (string, default `"0"`): camera index or video path.

**Response (example)**
```json
{
  "message": "Proctoring started!",
  "source": "0",
  "pid": 12345
}
```

### `POST /stop`
Stops the running proctoring subprocess if active.

**Response (example)**
```json
{
  "message": "Proctoring stopped!"
}
```

---

## How Detection Works (Integrated Pipeline)

For each frame:
1. Detect face bounding boxes.
2. Extract 68 facial landmarks.
3. Build eye masks and estimate gaze direction from pupil contour geometry.
4. Estimate head pose with `solvePnP` + nose projection angle.
5. Detect mouth opening using calibrated lip landmark distances.
6. Compute YCrCb/LUV histogram features and classify spoof probability.
7. Render overlays + print warnings only when state changes.

---

## Troubleshooting

- **No camera opened**: verify webcam permissions and index.
- **Model load error**: check `models/` paths and filenames.
- **API starts but no processing**: resolve `Proctor.py` vs `proctor.py` naming mismatch.
- **PyAudio install errors**: install platform-specific audio build tools/drivers.
- **Slow inference**: reduce frame resolution or use hardware acceleration.

---

## Limitations

- Multiple scripts are research/demo-style and not packaged as production modules.
- Some assets (models/weights/datasets) are external and must be added manually.
- Real-world robustness depends on lighting, camera quality, occlusion, and model quality.
- Audio module uses cloud speech recognition API behavior and may be network-sensitive.

---

## Future Improvements

- Unify naming and startup flow between API and integrated script.
- Add config file for model paths and thresholds.
- Add logging, event recording, and report generation.
- Add Docker support with reproducible setup.
- Add tests and CI checks for core pipeline modules.

---

## Disclaimer

This project is intended for educational/research purposes. If used in real assessment environments, ensure compliance with privacy laws, consent requirements, and institutional policies.
