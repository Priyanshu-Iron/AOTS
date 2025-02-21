from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from utils.camera import Camera
from utils.detection import Detection
from utils.tracking import Tracker
import cv2
import numpy as np
import time
import logging
import os

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize global objects
camera = Camera()
detection = Detection('models/yolov3-tiny.weights', 'models/yolov3-tiny.cfg', 'models/coco.names')
tracker = Tracker('GOTURN')
tracking_enabled = False
detected_objects = []

# Initialize camera at startup
try:
    camera.initialize()
    logger.info("Camera initialized at startup")
except Exception as e:
    logger.error(f"Failed to initialize camera at startup: {e}")

def process_frame(frame):
    """Process the frame by adding bounding boxes for detection or tracking."""
    global tracking_enabled, detected_objects
    if frame is None:
        return None
    if tracking_enabled:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{obj['label']} {obj['confidence']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_camera', methods=['POST'])
def initialize_camera():
    try:
        camera.initialize()
        logger.info("Camera reinitialized successfully")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Camera reinitialization error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/detect_objects', methods=['POST'])
def detect_objects_route():
    global detected_objects
    try:
        frame = camera.get_frame()
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Could not read frame'})
        detected_objects = detection.detect(frame)
        logger.info(f"Detected objects: {detected_objects}")
        return jsonify({'status': 'success', 'objects': detected_objects})
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracking_enabled, tracker
    try:
        data = request.get_json()
        if not data or 'bbox' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid request'})
        bbox = tuple(int(v) for v in data['bbox'])
        frame = camera.get_frame()
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Could not read frame'})
        tracker.initialize(frame, bbox)
        tracking_enabled = True
        logger.info("Tracking started")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Start tracking error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_enabled, tracker
    tracking_enabled = False
    tracker = Tracker('GOTURN')  # Reset tracker
    logger.info("Tracking stopped")
    return jsonify({'status': 'success'})

def generate_frames():
    """Generate frames for video streaming."""
    while True:
        if camera.cap is None or not camera.cap.isOpened():
            logger.warning("Camera not initialized, waiting...")
            time.sleep(0.1)
            # Generate a placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not initialized", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            frame = camera.get_frame()
            if frame is None:
                continue
            frame = process_frame(frame)
            if frame is None:
                continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)