from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
from utils.camera import Camera
import cv2
import numpy as np
import time
import logging
import os
from ultralytics import YOLO
import psutil
import pynvml

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NVIDIA Management Library for GPU stats (safe for non-NVIDIA systems)
try:
    pynvml.nvmlInit()
    gpu_available = True
    logger.info("NVIDIA GPU detected and initialized.")
except pynvml.NVMLError:
    gpu_available = False
    logger.info("No NVIDIA GPU detected or pynvml not installed. GPU stats will be unavailable.")

# Define Detection class for YOLOv8
class Detection:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            self.model = YOLO(model_path)
            self.model_name = 'YOLOv8n'
            logger.info(f"Object detection model loaded: {self.model_name} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def detect(self, frame, confidence_threshold=0.3):
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
            logger.info(f"Total detected objects: {len(detected_objects)} using {self.model_name}")
            return detected_objects
        except Exception as e:
            logger.error(f"Detection error with {self.model_name}: {e}")
            return []

# Define Tracker class with CSRT
class Tracker:
    def __init__(self):
        self.tracker = None
        self.bbox = None
        self.tracker_type = 'CSRT'
        logger.info(f"Tracking model initialized: {self.tracker_type}")

    def initialize(self, frame, bbox):
        self.bbox = bbox
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        logger.info(f"Tracker initialized with {self.tracker_type} at bbox {bbox}")

    def update(self, frame):
        if self.tracker is None:
            return False, self.bbox
        success, new_bbox = self.tracker.update(frame)
        if success:
            self.bbox = new_bbox
            logger.debug(f"Tracker {self.tracker_type} updated successfully to bbox {new_bbox}")
        else:
            logger.warning(f"Tracker {self.tracker_type} lost object")
        return success, self.bbox

# Memory for learned objects
class ObjectMemory:
    def __init__(self):
        self.objects = {}  # {id: {"label": str, "bbox": list, "last_seen": float}}

    def add_object(self, label, bbox):
        obj_id = f"{label}_{len(self.objects)}"
        self.objects[obj_id] = {"label": label, "bbox": bbox, "last_seen": time.time()}
        logger.info(f"Memorized object {obj_id} with bbox {bbox}")
        return obj_id

    def update_object(self, obj_id, bbox):
        if obj_id in self.objects:
            self.objects[obj_id]["bbox"] = bbox
            self.objects[obj_id]["last_seen"] = time.time()

    def get_similar(self, label, bbox, iou_threshold=0.5):
        for obj_id, obj in self.objects.items():
            if obj["label"] == label:
                iou = calculate_iou(bbox, obj["bbox"])
                if iou > iou_threshold:
                    return obj_id
        return None

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Initialize global objects
camera = Camera()
detection = Detection('yolov8n.pt')
tracker = Tracker()
memory = ObjectMemory()
tracking_enabled = False
detected_objects = []
current_obj_id = None

try:
    camera.initialize()
    logger.info("Camera initialized at startup")
except Exception as e:
    logger.error(f"Failed to initialize camera at startup: {e}")

def process_frame(frame):
    global tracking_enabled, detected_objects, current_obj_id
    if frame is None:
        return None
    if tracking_enabled and current_obj_id:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking {memory.objects[current_obj_id]['label']}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            memory.update_object(current_obj_id, bbox)
        else:
            tracking_enabled = False
            current_obj_id = None
            cv2.putText(frame, "Lost", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        detected_objects = detection.detect(frame)
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            similar_id = memory.get_similar(obj['label'], obj['bbox'])
            if similar_id:
                obj_id = similar_id
                memory.update_object(obj_id, obj['bbox'])
                cv2.putText(frame, f"Recognized {obj['label']}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                obj_id = memory.add_object(obj['label'], obj['bbox'])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{obj['label']} {obj['confidence']:.2f}", (x, y - 25),
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

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    try:
        if camera.cap is not None:
            camera.cap.release()
            camera.cap = None
            logger.info("Camera stopped successfully")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Camera stop error: {e}")
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
    global tracking_enabled, tracker, current_obj_id
    try:
        data = request.get_json()
        if not data or 'bbox' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid request'})
        bbox = tuple(int(v) for v in data['bbox'])
        frame = camera.get_frame()
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Could not read frame'})
        label = data.get('label', 'unknown')
        similar_id = memory.get_similar(label, bbox)
        if similar_id:
            current_obj_id = similar_id
        else:
            current_obj_id = memory.add_object(label, bbox)
        tracker.initialize(frame, bbox)
        tracking_enabled = True
        logger.info(f"Tracking started with {tracker.tracker_type} for {current_obj_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Start tracking error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_enabled, tracker, current_obj_id
    tracking_enabled = False
    tracker = Tracker()
    current_obj_id = None
    logger.info(f"Tracking stopped, reset to {tracker.tracker_type}")
    return jsonify({'status': 'success'})

@app.route('/tracker_info')
def tracker_info():
    status = 'Tracking' if tracking_enabled else 'Idle'
    return jsonify({
        'type': tracker.tracker_type,
        'status': status,
        'bbox': list(tracker.bbox) if tracker.bbox else None,
        'detection_model': detection.model_name,
        'memorized_objects': list(memory.objects.keys())
    })

@app.route('/system_stats')
def system_stats():
    stats = {'cpu_percent': psutil.cpu_percent(interval=1)}
    if gpu_available:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            stats['gpu_percent'] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats['gpu_mem_used'] = mem_info.used / 1024**2  # MB
            stats['gpu_mem_total'] = mem_info.total / 1024**2  # MB
        except pynvml.NVMLError as e:
            logger.error(f"GPU stats error: {e}")
            stats['gpu_percent'] = 0
            stats['gpu_mem_used'] = 0
            stats['gpu_mem_total'] = 0
    else:
        stats['gpu_percent'] = 0
        stats['gpu_mem_used'] = 0
        stats['gpu_mem_total'] = 0
    return jsonify(stats)

def generate_frames():
    while True:
        if camera.cap is None or not camera.cap.isOpened():
            logger.warning("Camera not initialized or stopped, waiting...")
            time.sleep(0.1)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera stopped", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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