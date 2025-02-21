from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from utils.camera import Camera
import cv2
import numpy as np
import time
import logging
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Detection class for YOLOv8
class Detection:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the YOLOv8 model."""
        try:
            self.model = YOLO(model_path)
            self.model_name = 'YOLOv8n'  # Store model name for display
            logger.info(f"Object detection model loaded: {self.model_name} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def detect(self, frame, confidence_threshold=0.3):
        """Detect objects in the given frame using YOLOv8 with smaller bounding boxes."""
        try:
            results = self.model(frame, conf=confidence_threshold)
            detected_objects = []
            for result in results:
                for box in result.boxes:
                    x, y, w, h = box.xywh[0].tolist()
                    scale_factor = 0.8
                    w, h = int(w * scale_factor), int(h * scale_factor)
                    x, y = int(x - w / 2), int(y - h / 2)
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    label = self.model.names[class_id]
                    detected_objects.append({
                        "label": label,
                        "bbox": [x, y, w, h],
                        "confidence": confidence
                    })
                    logger.debug(f"Detected: {label} at {x},{y},{w},{h} with confidence {confidence:.2f}")
            logger.info(f"Total detected objects: {len(detected_objects)} using {self.model_name}")
            return detected_objects
        except Exception as e:
            logger.error(f"Detection error with {self.model_name}: {e}")
            return []

# Define Tracker class with CSRT only
class Tracker:
    def __init__(self):
        """Initialize the CSRT tracker."""
        self.tracker = None
        self.bbox = None
        self.tracker_type = 'CSRT'
        logger.info(f"Tracking model initialized: {self.tracker_type}")

    def initialize(self, frame, bbox):
        """Initialize the tracker with the given bounding box."""
        self.bbox = bbox
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        logger.info(f"Tracker initialized with {self.tracker_type} at bbox {bbox}")

    def update(self, frame):
        """Update the tracker with a new frame and return the updated bounding box."""
        if self.tracker is None:
            return False, self.bbox
        success, new_bbox = self.tracker.update(frame)
        if success:
            self.bbox = new_bbox
            logger.debug(f"Tracker {self.tracker_type} updated successfully to bbox {new_bbox}")
        else:
            logger.warning(f"Tracker {self.tracker_type} lost object")
        return success, self.bbox

# Initialize global objects
camera = Camera()
detection = Detection('yolov8n.pt')
tracker = Tracker()
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
            scale_factor = 0.8
            w, h = int(w * scale_factor), int(h * scale_factor)
            x, y = int(x + (obj['bbox'][2] - w) / 2), int(y + (obj['bbox'][3] - h) / 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{obj['label']} {obj['confidence']:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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
        logger.info(f"Tracking started with {tracker.tracker_type}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Start tracking error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_enabled, tracker
    tracking_enabled = False
    tracker = Tracker()
    logger.info(f"Tracking stopped, reset to {tracker.tracker_type}")
    return jsonify({'status': 'success'})

@app.route('/tracker_info')
def tracker_info():
    """Return current tracker information."""
    status = 'Tracking' if tracking_enabled else 'Idle'
    return jsonify({
        'type': tracker.tracker_type,
        'status': status,
        'bbox': list(tracker.bbox) if tracker.bbox else None,
        'detection_model': detection.model_name
    })

def generate_frames():
    """Generate frames for video streaming."""
    while True:
        if camera.cap is None or not camera.cap.isOpened():
            logger.warning("Camera not initialized, waiting...")
            time.sleep(0.1)
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