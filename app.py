from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import logging
import os
import psutil
import pynvml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from collections import deque
from ultralytics import YOLO

# Import custom classes from utils folder
from utils.camera import Camera
from utils.detection import Detection
from utils.tracking import Tracker

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NVIDIA Management Library for GPU stats
try:
    pynvml.nvmlInit()
    gpu_available = True
    logger.info("NVIDIA GPU detected and initialized.")
except pynvml.NVMLError:
    gpu_available = False
    logger.info("No NVIDIA GPU detected or pynvml not installed.")

class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FeatureExtractor, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.backbone.fc = nn.Linear(512, embedding_dim)
        self.normalize = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.normalize(features)

class MLObjectMemory:
    def __init__(self, max_objects=50, feature_dim=128):
        self.feature_extractor = FeatureExtractor(embedding_dim=feature_dim)
        self.feature_extractor.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.objects = {}
        self.feature_history = {}
        self.max_objects = max_objects
        self.max_lost_frames = 30
        self.feature_history_length = 5
        self.similarity_threshold = 0.75

    def extract_features(self, frame, bbox):
        try:
            x, y, w, h = map(int, bbox)
            roi = frame[max(0, y):min(frame.shape[0], y + h),
                       max(0, x):min(frame.shape[1], x + w)]
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_tensor = self.transform(roi_pil).unsqueeze(0)
            with torch.no_grad():
                features = self.feature_extractor(roi_tensor)
            return features.squeeze().numpy()
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def update_feature_history(self, obj_id, features):
        if obj_id not in self.feature_history:
            self.feature_history[obj_id] = deque(maxlen=self.feature_history_length)
        self.feature_history[obj_id].append(features)

    def compute_similarity(self, features1, features2):
        return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))

    def get_average_features(self, obj_id):
        if obj_id in self.feature_history:
            features_list = list(self.feature_history[obj_id])
            return np.mean(features_list, axis=0)
        return None

    def add_object(self, frame, label, bbox):
        features = self.extract_features(frame, bbox)
        if features is None:
            return None
        obj_id = f"{label}_{len(self.objects)}"
        self.objects[obj_id] = {
            "label": label,
            "bbox": bbox,
            "features": features,
            "last_seen": time.time(),
            "lost_count": 0
        }
        self.update_feature_history(obj_id, features)
        logger.info(f"Added new object {obj_id} with features")
        return obj_id

    def find_best_match(self, frame, label, bbox):
        current_features = self.extract_features(frame, bbox)
        if current_features is None:
            return None, None
        best_match = None
        best_similarity = self.similarity_threshold
        for obj_id, obj in self.objects.items():
            if obj["label"] == label and obj["lost_count"] < self.max_lost_frames:
                avg_features = self.get_average_features(obj_id)
                if avg_features is not None:
                    similarity = self.compute_similarity(current_features, avg_features)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = obj_id
        return best_match, current_features

    def update_object(self, obj_id, frame, bbox):
        # Skip feature extraction on update to save time
        self.objects[obj_id].update({
            "bbox": bbox,
            "last_seen": time.time(),
            "lost_count": 0
        })

    def cleanup_old_objects(self):
        self.objects = {
            obj_id: obj for obj_id, obj in self.objects.items()
            if obj["lost_count"] < self.max_lost_frames
        }
        self.feature_history = {
            obj_id: hist for obj_id, hist in self.feature_history.items()
            if obj_id in self.objects
        }

    def get_similar(self, label, bbox):
        iou_threshold = 0.3
        for obj_id, obj in self.objects.items():
            if obj["label"] == label and obj["lost_count"] < self.max_lost_frames:
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
tracker = Tracker(tracker_type='CSRT')  # Switch to CSRT for better performance
memory = MLObjectMemory()
tracking_enabled = False
detected_objects = []
current_obj_id = None

def process_frame(frame, frame_count=0):
    global tracking_enabled, detected_objects, current_obj_id, memory
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        logger.error("Invalid frame received in process_frame")
        return None

    # Update lost counts
    for obj_id in memory.objects:
        memory.objects[obj_id]["lost_count"] += 1

    # Process tracking
    if tracking_enabled and current_obj_id:
        try:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tracking {memory.objects[current_obj_id]['label']}",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                memory.update_object(current_obj_id, frame, bbox)
            else:
                tracking_enabled = False
                current_obj_id = None
                cv2.putText(frame, "Lost", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            logger.error(f"Tracker update failed: {e}")
            tracking_enabled = False
            current_obj_id = None
            cv2.putText(frame, "Tracking Error", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Detect objects every 5 frames
    DETECTION_INTERVAL = 5
    if frame_count % DETECTION_INTERVAL == 0:
        detected_objects = detection.detect(frame)
        for obj in detected_objects:
            bbox = obj['bbox']
            label = obj['label']
            match_id, features = memory.find_best_match(frame, label, bbox)
            if match_id:
                memory.update_object(match_id, frame, bbox)
                color = (0, 255, 255)  # Yellow for recognized
                if not tracking_enabled:
                    current_obj_id = match_id
                    tracker.initialize(frame, tuple(bbox))
                    tracking_enabled = True
                    color = (0, 255, 0)  # Green for tracked
            else:
                obj_id = memory.add_object(frame, label, bbox)
                if obj_id:
                    color = (255, 0, 0)  # Blue for new
                else:
                    continue
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {obj['confidence']:.2f}",
                        (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    memory.cleanup_old_objects()
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
            current_obj_id = memory.add_object(frame, label, bbox)
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
    tracker = Tracker(tracker_type='CSRT')
    current_obj_id = None
    logger.info(f"Tracking stopped, reset to {tracker.tracker_type}")
    return jsonify({'status': 'success'})

@app.route('/tracker_info')
def tracker_info():
    return jsonify({
        'type': tracker.tracker_type,
        'status': 'Tracking' if tracking_enabled else 'Idle',
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
            stats['gpu_mem_used'] = mem_info.used / 1024**2
            stats['gpu_mem_total'] = mem_info.total / 1024**2
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
    frame_count = 0
    while True:
        start_time = time.time()
        if camera.cap is None or not camera.cap.isOpened():
            logger.warning("Camera not initialized or stopped")
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera stopped", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            frame = camera.get_frame()
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.warning("Failed to get valid frame from camera")
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera feed", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                frame = process_frame(frame, frame_count)
                frame_count += 1
                if frame is None:
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(frame, "Processing error", (20, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        logger.info(f"FPS: {fps:.2f}")

        try:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                logger.error("Failed to encode frame")
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Frame encoding/streaming error: {e}")
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        camera.initialize()
        camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
        camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        logger.info("Camera initialized at startup with 320x240 resolution")
    except Exception as e:
        logger.error(f"Failed to initialize camera at startup: {e}")

    app.run(debug=True, host='0.0.0.0', port=5500)