# Sleep Detection using MediaPipe & OpenCV

This Python project detects drowsiness or sleep in real-time using Eye Aspect Ratio (EAR) from face landmarks with MediaPipe.

## 🛠️ Requirements

Install using pip:

```bash
pip install opencv-python mediapipe numpy
```

## ▶️ How to Run
 
```bash
python sleep_detector.py
```

The webcam will open, and you'll see:
- **AWAKE** (when eyes are open)
- **SLEEPING** (when eyes are closed for a few seconds)

Press **ESC** to exit.
