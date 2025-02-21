import cv2
import os

class Tracker:
    def __init__(self, tracker_type='GOTURN'):
        """Initialize the tracker with the specified type."""
        self.tracker_type = tracker_type
        self.tracker = None
        self.bbox = None

    def initialize(self, frame, bbox):
        """Initialize the tracker with the given bounding box."""
        self.bbox = bbox
        if self.tracker_type == 'CSRT':
            self.tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == 'GOTURN':
            prototxt_path = 'goturn/goturn.prototxt'
            caffemodel_path = 'goturn/goturn.caffemodel'
            if not (os.path.exists(prototxt_path) and os.path.exists(caffemodel_path)):
                raise FileNotFoundError("GOTURN model files not found")
            params = cv2.TrackerGOTURN_Params()
            params.modelTxt = prototxt_path
            params.modelBin = caffemodel_path
            self.tracker = cv2.TrackerGOTURN.create(params)
        else:
            raise ValueError("Invalid tracker type")
        self.tracker.init(frame, bbox)

    def update(self, frame):
        """Update the tracker with a new frame and return the updated bounding box."""
        if self.tracker is None:
            return False, self.bbox
        success, new_bbox = self.tracker.update(frame)
        if success:
            self.bbox = new_bbox
        return success, self.bbox