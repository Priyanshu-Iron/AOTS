import cv2
import platform

class Camera:
    def __init__(self):
        self.cap = None

    def initialize(self):
        if self.cap is not None:
            self.cap.release()
        if platform.system() == 'Darwin':  # macOS
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Adjusted in app.py
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()