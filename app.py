from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import logging
import traceback
import platform
import os

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
kalman = None
source_type = None
is_paused = False
last_frame = None
drone_pos = {'x': 0, 'y': 0, 'z': 10}

# GOTURN model paths
GOTURN_PROTO = os.path.join(os.path.dirname(__file__), 'goturn.prototxt')
GOTURN_MODEL = os.path.join(os.path.dirname(__file__), 'goturn.caffemodel')

def init_camera():
    global camera, source_type, is_paused
    camera_instance = None
    try:
        if camera is not None:
            camera.release()
            camera = None
        
        camera_instance = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        time.sleep(3)
        
        if not camera_instance.isOpened():
            raise RuntimeError("Could not open camera")
        
        camera_instance.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        timeout = 5
        while time.time() - start_time < timeout:
            ret, frame = camera_instance.read()
            if ret and frame is not None:
                logger.info(f"Camera initialized successfully with AVFoundation. Frame shape: {frame.shape}")
                camera = camera_instance
                source_type = 'drone_sim'
                is_paused = False
                return True
            time.sleep(0.1)
        
        raise RuntimeError("Could not read from camera")
    except Exception as e:
        logger.error(f"Camera initialization error: {str(e)}")
        if camera_instance is not None:
            camera_instance.release()
        return False
    finally:
        if 'camera_instance' in locals() and camera_instance is not None and camera is None:
            camera_instance.release()

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

class BasicTracker:
    def __init__(self):
        self.bbox = None
        self.template = None
        self.method = cv2.TM_CCOEFF_NORMED
        self.confidence_threshold = 0.7  # Adjust lower if needed
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
            self.template = preprocess_frame(frame)[y:y+h, x:x+w].copy()
            logger.debug(f"BasicTracker initialized with template shape: {self.template.shape}")
            return True
        except Exception as e:
            logger.error(f"BasicTracker init error: {str(e)}")
            return False
    
    def update(self, frame):
        if self.template is None or self.bbox is None:
            logger.warning("BasicTracker update failed: No template or bbox")
            return False, self.bbox
        x, y, w, h = [int(v) for v in self.bbox]
        search_factor = 3.0  # Increased for better tracking of moving objects
        search_x = max(0, x - int(w * search_factor / 2))
        search_y = max(0, y - int(h * search_factor / 2))
        search_w = min(frame.shape[1] - search_x, int(w * search_factor))
        search_h = min(frame.shape[0] - search_y, int(h * search_factor))
        if search_w <= 0 or search_h <= 0:
            logger.warning("BasicTracker update failed: Invalid search region dimensions")
            return False, self.bbox
        search_region = preprocess_frame(frame)[search_y:search_y+search_h, search_x:search_x+search_w]
        if search_region.shape[0] < self.template.shape[0] or search_region.shape[1] < self.template.shape[1]:
            logger.warning("BasicTracker update failed: Search region smaller than template")
            return False, self.bbox
        try:
            res = cv2.matchTemplate(search_region, self.template, self.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            logger.debug(f"BasicTracker match confidence: {max_val}")
            if max_val < self.confidence_threshold:
                logger.warning(f"BasicTracker update failed: Low confidence match: {max_val}")
                return False, self.bbox
            new_x = search_x + max_loc[0]
            new_y = search_y + max_loc[1]
            self.bbox = (new_x, new_y, w, h)
            new_template = preprocess_frame(frame)[new_y:new_y+h, new_x:new_x+w].copy()
            alpha = 0.1
            if new_template.shape == self.template.shape:
                self.template = cv2.addWeighted(self.template, 1-alpha, new_template, alpha, 0)
            return True, self.bbox
        except Exception as e:
            logger.error(f"Template matching error: {e}")
            return False, self.bbox

def init_kalman():
    global kalman
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

def init_tracker(frame, bbox):
    try:
        # GOTURN Initialization
        if not os.path.exists(GOTURN_PROTO) or not os.path.exists(GOTURN_MODEL):
            logger.error(f"GOTURN model files missing: {GOTURN_PROTO}, {GOTURN_MODEL}")
        elif not hasattr(cv2, 'TrackerGOTURN_create'):
            logger.error("GOTURN tracker not available in OpenCV build")
        else:
            logger.info("Attempting to initialize GOTURN tracker")
            tracker_instance = cv2.TrackerGOTURN_create()
            processed_frame = preprocess_frame(frame)
            logger.debug(f"Initializing GOTURN with frame shape: {processed_frame.shape}, bbox: {bbox}")
            try:
                success = tracker_instance.init(processed_frame, bbox)
                if success:
                    logger.info("Successfully initialized GOTURN tracker")
                    init_kalman()
                    kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
                    return tracker_instance
                else:
                    logger.error("GOTURN tracker initialization failed silently")
            except Exception as e:
                logger.error(f"GOTURN initialization raised exception: {str(e)}")
                traceback.print_exc()

        # Fallback to CSRT
        logger.info("Falling back to CSRT tracker")
        if hasattr(cv2, 'TrackerCSRT_create'):
            tracker_instance = cv2.TrackerCSRT_create()
            processed_frame = preprocess_frame(frame)
            logger.debug(f"Initializing CSRT with frame shape: {processed_frame.shape}, bbox: {bbox}")
            try:
                success = tracker_instance.init(processed_frame, bbox)
                if success:
                    logger.info("Successfully initialized CSRT tracker")
                    init_kalman()
                    kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
                    return tracker_instance
                else:
                    logger.error("CSRT tracker initialization failed silently")
            except Exception as e:
                logger.error(f"CSRT initialization raised exception: {str(e)}")
                traceback.print_exc()
        else:
            logger.error("CSRT tracker not available in OpenCV build")

        # Final Fallback to BasicTracker
        logger.info("Falling back to BasicTracker")
        basic_tracker = BasicTracker()
        if basic_tracker.init(preprocess_frame(frame), bbox):
            logger.info("Successfully initialized BasicTracker")
            init_kalman()
            kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
            return basic_tracker
        else:
            logger.error("Failed to initialize BasicTracker")
            return None
    except Exception as e:
        logger.error(f"Unexpected error in init_tracker: {str(e)}")
        traceback.print_exc()
        return None

def simulate_drone_control(bbox, frame_width, frame_height):
    global drone_pos
    if not tracking_enabled:
        return
    
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    delta_x = center_x - frame_center_x
    delta_y = center_y - frame_center_y
    
    drone_pos['x'] += delta_x * 0.01
    drone_pos['y'] += delta_y * 0.01
    drone_pos['z'] -= delta_y * 0.005
    
    logger.debug(f"Simulated drone position: x={drone_pos['x']:.2f}, y={drone_pos['y']:.2f}, z={drone_pos['z']:.2f}")

def process_frame(frame):
    global tracker, bbox, tracking_enabled, kalman
    try:
        if frame is None:
            logger.warning("Frame is None in process_frame")
            return frame
        if tracking_enabled and bbox is not None and tracker is not None:
            processed_frame = preprocess_frame(frame)
            success, new_bbox = tracker.update(processed_frame)
            logger.debug(f"Tracker update: Success={success}, New bbox={new_bbox}")
            if success:
                x, y, w, h = [int(v) for v in new_bbox]
                kalman.predict()
                measurement = np.array([[np.float32(x + w/2)], [np.float32(y + h/2)]])
                kalman.correct(measurement)
                smoothed_center = kalman.statePost[:2].flatten()
                smoothed_x = int(smoothed_center[0] - w/2)
                smoothed_y = int(smoothed_center[1] - h/2)
                bbox = (smoothed_x, smoothed_y, w, h)
                
                cv2.rectangle(frame, (smoothed_x, smoothed_y), (smoothed_x + w, smoothed_y + h), (0, 255, 0), 2)
                center_x = smoothed_x + w // 2
                center_y = smoothed_y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Object at: ({center_x}, {center_y})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                simulate_drone_control(bbox, frame.shape[1], frame.shape[0])
            else:
                logger.warning("Tracking failed in process_frame")
                cv2.putText(frame, "Tracking failed", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_drone_sim', methods=['POST'])
def initialize_drone_sim():
    try:
        if init_camera():
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Failed to initialize drone simulation'})
    except Exception as e:
        logger.error(f"Drone sim init route error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused, last_frame
    try:
        data = request.get_json()
        if data is None or 'pause' not in data:
            logger.error("Invalid or missing JSON data in toggle_pause request")
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON data'})
        is_paused = data['pause']
        logger.info(f"Drone feed {'paused' if is_paused else 'resumed'}")
        return jsonify({'status': 'success', 'is_paused': is_paused})
    except Exception as e:
        logger.error(f"Toggle pause error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

def generate_frames():
    global camera, source_type, is_paused, last_frame
    retry_count = 0
    max_retries = 5
    while True:
        try:
            if source_type == 'drone_sim':
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
                if is_paused and last_frame is not None:
                    frame = last_frame.copy()
                else:
                    ret, frame = camera.read()
                    if not ret or frame is None:
                        logger.warning("Failed to read frame from camera, attempting to recover...")
                        camera.release()
                        camera = None
                        time.sleep(1)
                        continue
            else:
                time.sleep(0.5)
                continue
            
            frame = process_frame(frame)
            if not is_paused:
                last_frame = frame.copy()
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
    global tracker, bbox, tracking_enabled, camera, source_type, is_paused
    try:
        if source_type != 'drone_sim' or (camera is None or not camera.isOpened()):
            logger.error("Drone simulation not initialized or camera not opened")
            return jsonify({'status': 'error', 'message': 'Drone simulation not initialized'})
        
        data = request.get_json()
        if data is None or 'bbox' not in data:
            logger.error("Invalid or missing JSON data in start_tracking request")
            return jsonify({'status': 'error', 'message': 'Invalid or missing JSON data'})
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
        is_paused = False
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
    global camera, tracking_enabled, source_type, is_paused, drone_pos
    try:
        source_status = "initialized" if camera is not None and camera.isOpened() else "not initialized"
        return jsonify({
            'status': 'success',
            'source_type': source_type if source_type else "none",
            'source_status': source_status,
            'tracking': tracking_enabled,
            'is_paused': is_paused,
            'drone_pos': drone_pos,
            'opencv_version': cv2.__version__
        })
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    try:
        if not os.path.exists(GOTURN_PROTO) or not os.path.exists(GOTURN_MODEL):
            logger.error(f"GOTURN model files missing: {GOTURN_PROTO}, {GOTURN_MODEL}")
        app.run(debug=True, port=5500, host='0.0.0.0')
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    finally:
        if camera is not None:
            camera.release()
            logger.info("Camera released on shutdown")