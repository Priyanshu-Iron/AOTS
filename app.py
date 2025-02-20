from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import logging
import traceback
import platform

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
tracker = None
tracking_enabled = False
bbox = None
camera = None

def init_camera():
    """Initialize the camera with multiple backend attempts and diagnostics."""
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
        
        os_name = platform.system()
        logger.info(f"Detected OS: {os_name}")
        
        backends = [
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_V4L2, "V4L2 (Linux)"),
            (cv2.CAP_AVFOUNDATION, "AVFoundation (Mac)")
        ]
        
        for backend, backend_name in backends:
            try:
                logger.info(f"Attempting to open camera with {backend_name} backend")
                camera = cv2.VideoCapture(0, backend)
                time.sleep(3)
                
                if not camera.isOpened():
                    logger.warning(f"Failed to open camera with {backend_name}")
                    camera.release()
                    continue
                
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                start_time = time.time()
                timeout = 5
                while time.time() - start_time < timeout:
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        logger.info(f"Camera initialized successfully with {backend_name}. Frame shape: {frame.shape}")
                        return True
                    time.sleep(0.1)
                
                logger.warning(f"Test read failed with {backend_name}")
                camera.release()
            except Exception as e:
                logger.error(f"Error with {backend_name} backend: {str(e)}")
                if camera is not None:
                    camera.release()
        
        raise RuntimeError("No suitable camera backend found")
    except Exception as e:
        logger.error(f"Camera initialization error: {str(e)}")
        if camera is not None:
            camera.release()
            camera = None
        return False

class BasicTracker:
    def __init__(self):
        self.bbox = None
        self.template = None
        self.method = cv2.TM_CCOEFF_NORMED
        logger.info("Using BasicTracker fallback")
    
    def init(self, frame, bbox):
        if frame is None:
            logger.error("BasicTracker init error: Frame is None")
            return False
        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            logger.error(f"BasicTracker init error: Invalid bbox dimensions - w:{w}, h:{h}")
            return False
        try:
            self.bbox = bbox
            self.template = frame[y:y+h, x:x+w].copy()
            logger.debug(f"BasicTracker initialized with template shape: {self.template.shape}")
            return True
        except Exception as e:
            logger.error(f"BasicTracker init error: {str(e)}")
            return False
    
    def update(self, frame):
        if self.template is None or self.bbox is None:
            return False, self.bbox
        x, y, w, h = [int(v) for v in self.bbox]
        search_factor = 2
        search_x = max(0, x - w//2)
        search_y = max(0, y - h//2)
        search_w = min(frame.shape[1] - search_x, w*search_factor)
        search_h = min(frame.shape[0] - search_y, h*search_factor)
        if search_w <= 0 or search_h <= 0:
            return False, self.bbox
        search_region = frame[search_y:search_y+search_h, search_x:search_x+search_w]
        if search_region.shape[0] < self.template.shape[0] or search_region.shape[1] < self.template.shape[1]:
            return False, self.bbox
        try:
            res = cv2.matchTemplate(search_region, self.template, self.method)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            new_x = search_x + max_loc[0]
            new_y = search_y + max_loc[1]
            self.bbox = (new_x, new_y, w, h)
            new_template = frame[new_y:new_y+h, new_x:new_x+w].copy()
            alpha = 0.2
            if new_template.shape == self.template.shape:
                self.template = cv2.addWeighted(self.template, 1-alpha, new_template, alpha, 0)
            return True, self.bbox
        except Exception as e:
            logger.error(f"Template matching error: {e}")
            return False, self.bbox

def init_tracker(frame, bbox):
    """Initialize tracker with multiple attempts and diagnostics."""
    try:
        opencv_version = cv2.__version__.split('.')
        major_version = int(opencv_version[0])
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        if major_version >= 4:
            tracker_types = ['KCF', 'CSRT', 'MOSSE']
            for tracker_type in tracker_types:
                try:
                    tracker_create_func = getattr(cv2, f'Tracker{tracker_type}_create', None)
                    if not tracker_create_func:
                        logger.warning(f"{tracker_type} tracker not available in this OpenCV build")
                        continue
                    logger.info(f"Creating tracker using {tracker_type}")
                    tracker_instance = tracker_create_func()
                    success = tracker_instance.init(frame, bbox)
                    if success:
                        logger.info(f"Successfully initialized {tracker_type} tracker")
                        return tracker_instance
                    else:
                        logger.warning(f"Failed to initialize {tracker_type} tracker")
                except Exception as e:
                    logger.error(f"Error creating/initializing {tracker_type} tracker: {str(e)}")
                    continue
            
            logger.info("All advanced trackers failed, falling back to BasicTracker")
            basic_tracker = BasicTracker()
            if basic_tracker.init(frame, bbox):
                return basic_tracker
            else:
                raise RuntimeError("BasicTracker initialization failed")
        else:
            logger.info("Using legacy OpenCV 3.x tracker API")
            tracker = cv2.Tracker_create("KCF")
            if tracker.init(frame, bbox):
                return tracker
            raise RuntimeError("Legacy KCF tracker initialization failed")
    except Exception as e:
        logger.error(f"Tracker initialization error: {str(e)}")
        traceback.print_exc()
        return None

def process_frame(frame):
    global tracker, bbox, tracking_enabled
    try:
        if frame is None:
            return frame
        if tracking_enabled and bbox is not None and tracker is not None:
            success, new_bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in new_bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Object at: ({center_x}, {center_y})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failed", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_camera', methods=['POST'])
def initialize_camera():
    try:
        if init_camera():
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Failed to initialize camera'})
    except Exception as e:
        logger.error(f"Camera init route error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

def generate_frames():
    global camera
    retry_count = 0
    max_retries = 5
    while True:
        try:
            if camera is None or not camera.isOpened():
                if retry_count >= max_retries:
                    logger.error("Max retries reached, stopping frame generation")
                    break
                if not init_camera():
                    logger.error("Failed to reinitialize camera")
                    retry_count += 1
                    time.sleep(1)
                    continue
                retry_count = 0
            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame, attempting to recover...")
                camera.release()
                camera = None
                time.sleep(1)
                continue
            frame = process_frame(frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame to JPEG")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)
        except Exception as e:
            logger.error(f"Frame generation error: {str(e)}")
            time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracker, bbox, tracking_enabled, camera
    try:
        if camera is None or not camera.isOpened():
            logger.error("Camera not initialized or not opened")
            return jsonify({'status': 'error', 'message': 'Camera not initialized'})
        if not request.is_json:
            logger.error("Invalid content type, expected JSON")
            return jsonify({'status': 'error', 'message': 'Invalid content type, expected JSON'})
        data = request.get_json()
        if not data or 'bbox' not in data:
            logger.error("Missing bbox data in request")
            return jsonify({'status': 'error', 'message': 'Invalid bbox data'})
        bbox_data = data['bbox']
        logger.debug(f"Received bbox data: {bbox_data}")
        if not isinstance(bbox_data, list) or len(bbox_data) != 4:
            logger.error(f"Invalid bbox format: {bbox_data}")
            return jsonify({'status': 'error', 'message': 'Invalid bbox format. Expected list of 4 numbers.'})
        MIN_SIZE = 10
        x, y, w, h = [max(0, int(float(value))) for value in bbox_data]
        w = max(w, MIN_SIZE)
        h = max(h, MIN_SIZE)
        bbox = (x, y, w, h)
        logger.info(f"Adjusted bbox: {bbox}")
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.error("Could not read frame from camera")
            return jsonify({'status': 'error', 'message': 'Could not read frame'})
        if x + w > frame.shape[1] or y + h > frame.shape[0]:
            logger.error("Bounding box exceeds frame dimensions")
            return jsonify({'status': 'error', 'message': 'Bounding box exceeds frame dimensions'})
        logger.debug(f"Frame shape before tracker init: {frame.shape}")
        tracker = init_tracker(frame, bbox)
        if tracker is None:
            logger.error("Tracker initialization returned None")
            return jsonify({'status': 'error', 'message': 'Failed to initialize tracker'})
        tracking_enabled = True
        logger.info("Tracking successfully initialized")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Unexpected error in start_tracking: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_enabled
    try:
        tracking_enabled = False
        logger.info("Tracking stopped")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error stopping tracking: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/check_status', methods=['GET'])
def check_status():
    global camera, tracking_enabled
    try:
        camera_status = "initialized" if camera is not None and camera.isOpened() else "not initialized"
        return jsonify({
            'status': 'success',
            'camera': camera_status,
            'tracking': tracking_enabled,
            'opencv_version': cv2.__version__
        })
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5500, host='0.0.0.0')
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        if camera is not None:
            camera.release()
            logger.info("Camera released on shutdown")