from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Detection:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            self.model = YOLO(model_path)
            self.model_name = model_path
            logger.info(f"YOLOv8 model loaded successfully from {model_path}")
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
                    x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    label = self.model.names[class_id]
                    detected_objects.append({
                        "label": label,
                        "bbox": [x, y, w, h],
                        "confidence": confidence
                    })
                    logger.debug(f"Detected: {label} at {x},{y},{w},{h} with confidence {confidence:.2f}")
            logger.info(f"Total detected objects: {len(detected_objects)}")
            return detected_objects
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []