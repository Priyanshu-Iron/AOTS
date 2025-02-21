import cv2
import platform

class Camera:
    def __init__(self):
        self.cap = None

    def initialize(self):
        """Initialize the camera (webcam for now)."""
        if self.cap is not None:
            self.cap.release()
        if platform.system() == 'Darwin':  # macOS
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_frame(self):
        """Capture and return a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()