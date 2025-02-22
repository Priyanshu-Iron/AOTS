# 🛰️ Autonomous Object Tracking System (AOTS)

The Autonomous Object Tracking System (AOTS) is a real-time object detection and tracking application built with Flask, OpenCV, and YOLOv8. It uses a webcam feed to detect objects (e.g., people, cars) with YOLOv8 Nano, tracks them using the CSRT tracker from OpenCV, and maintains a simple memory of recognized objects based on IoU (Intersection over Union) similarity. The system provides a web interface to view the video feed, control the camera, and monitor system stats like CPU/GPU usage.

## 🚀 Features
- **🎯 Object Detection**: Powered by YOLOv8 Nano for fast and accurate detection.
- **📌 Object Tracking**: Uses OpenCV's CSRT tracker to follow detected objects.
- **🧠 Object Memory**: Remembers objects using IoU-based similarity matching.
- **🌐 Web Interface**: Flask-based UI with real-time video feed, controls, and stats.
- **📊 Performance Monitoring**: Displays FPS, CPU usage, and GPU usage (if available).
- **⚡ Optimized for Speed**: Runs at 320x240 resolution for higher FPS.

## 🎥 In This Phase
I am currently playing a video in front of the webcam to replicate as a drone. Its main use is in defense to track objects from a drone, and once detected, the drone can track the object without a pilot automatically. 🛸

## 🔧 Requirements
- **🐍 Python 3.8+**
- **💻 Operating System**: macOS, Windows, or Linux
- **📷 Hardware**: Webcam (required), NVIDIA GPU (optional for GPU acceleration)

### 📦 Dependencies
Install the required Python packages:
```bash
pip install flask flask-cors opencv-python numpy ultralytics psutil pynvml torch torchvision pillow
```

- `flask` & `flask-cors`: Web framework and CORS support.
- `opencv-python`: Image processing and tracking.
- `numpy`: Array operations.
- `ultralytics`: YOLOv8 for object detection.
- `psutil`: CPU usage monitoring.
- `pynvml`: GPU usage monitoring (optional; requires NVIDIA GPU).
- `torch` & `torchvision`: PyTorch dependencies (used indirectly by YOLOv8).
- `pillow`: Image handling.

## 📂 Project Structure
```
AOTS/
│
├── app.py              # Main Flask application
├── utils/
│   ├── camera.py       # Camera handling class
│   ├── detection.py    # YOLOv8 detection class (not used in optimized version)
│   └── tracking.py     # CSRT tracker class (not used in optimized version)
├── templates/
│   └── index.html      # Web interface
└── README.md           # This file
```

Note: `detection.py` and `tracking.py` are included but not used in the optimized `app.py`, which defines these classes inline for simplicity.

## ⚙️ Setup
1. **🔽 Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd AOTS
   ```

2. **📥 Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install the packages listed above.

3. **📷 Ensure Webcam Access**:
   - Connect a webcam to your system.
   - On macOS/Windows, the app uses the default camera (index 0).

4. **⚡ (Optional) NVIDIA GPU Setup**:
   - Install NVIDIA drivers and CUDA if you have a compatible GPU.
   - Ensure `pynvml` detects your GPU (`gpu_available` will log as `True`).

## ▶️ Running the Application
1. **Start the Flask Server**:
   ```bash
   python app.py
   ```
   - The app runs on `http://0.0.0.0:5500` by default.
   - Logs will display initialization status (e.g., camera, GPU detection).

2. **🌍 Access the Web Interface**:
   - Open a browser and navigate to `http://127.0.0.1:5500`.
   - You’ll see the video feed and control buttons.

## 🛠️ Usage
### 🎛️ Web Interface
- **📷 Initialize Camera**: Starts the webcam feed.
- **⏹️ Stop Camera**: Stops the webcam feed.
- **🔍 Detect Objects**: Runs YOLOv8 detection on the current frame (manual trigger).
- **📌 Start Tracking**: Begins tracking a selected object.
- **🚫 Stop Tracking**: Stops tracking and resumes detection.
- **📺 Video Feed**: Displays real-time video with bounding boxes (blue for new, yellow for recognized, green for tracked).
- **ℹ️ Tracker Info**: Shows model details, status, and memorized objects.
- **📊 System Stats**: Graphs CPU and GPU usage.

### ⚡ How It Works
1. **📸 Detection**: When not tracking, YOLOv8 Nano detects objects in every frame.
2. **🎯 Tracking**: Once started, CSRT tracks a selected object until lost.
3. **🧠 Memory**: Objects are remembered and matched using IoU; recognized objects are labeled accordingly.
4. **⚡ Performance**: FPS is displayed on the feed and logged for monitoring.

## 🚀 Performance
- **⚡ FPS**: ~20-40 FPS on typical hardware (varies with CPU/GPU, webcam speed).
- **🖼️ Resolution**: 320x240 for optimized performance (adjustable in `app.py`).
- **🔍 Detection**: YOLOv8 Nano inference takes ~50-70ms per frame (hardware-dependent).

To improve FPS further:
- Increase `DETECTION_INTERVAL` in `process_frame` (e.g., detect every 5 frames).
- Use a GPU with CUDA-enabled OpenCV and YOLOv8 (`self.model = YOLO('yolov8n.pt', device='cuda')`).

## ❓ Troubleshooting
- **🚫 Camera Fails to Initialize**: Check webcam connection and permissions.
- **🐌 Low FPS**: Reduce resolution further (e.g., 160x120) or use a GPU.
- **⚠️ YOLOv8 Errors**: Ensure `ultralytics` is installed and `yolov8n.pt` is downloaded (automatically on first run).
- **📉 GPU Stats Missing**: Normal if no NVIDIA GPU or `pynvml` fails; stats will show as 0.

## 🤝 Contributing
Feel free to fork this repository, submit pull requests, or open issues for bugs/features. Potential enhancements:
- 🏆 Multi-object tracking.
- ⏳ Configurable detection intervals.
- 📡 Support for other YOLOv8 models (e.g., Small, Medium).

## 📜 License
This project is unlicensed (public domain). Use it freely!

## 🙏 Acknowledgments
- 🎯 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model.
- 🔬 [OpenCV](https://opencv.org/) for tracking and image processing.
- 🌐 [Flask](https://flask.palletsprojects.com/) for the web framework.

