import cv2
import numpy as np
import os


def create_tracker(tracker_type):
    """
    Create tracker based on tracker type
    """
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'GOTURN':
        # Check for GOTURN model files
        success, caffemodel_path, prototxt_path = check_goturn_model()
        if not success:
            return None

        params = cv2.TrackerGOTURN_Params()
        params.modelTxt = prototxt_path
        params.modelBin = caffemodel_path
        return cv2.TrackerGOTURN.create(params)


def check_goturn_model():
    """
    Check if GOTURN model files exist
    """
    model_folder = "goturn"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    caffemodel_path = os.path.join(os.path.dirname(__file__), 'goturn.caffemodel')
    prototxt_path = os.path.join(os.path.dirname(__file__), 'goturn.prototxt')

    if not (os.path.exists(caffemodel_path) and os.path.exists(prototxt_path)):
        print(f"GOTURN model files not found in {model_folder}")
        return False, None, None
    return True, caffemodel_path, prototxt_path


def track_object(video_source=0, tracker_type='CSRT'):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Couldn't open video source")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't read video source")
        return

    # Select ROI
    print(f"Select object to track with {tracker_type} and press ENTER")
    bbox = cv2.selectROI(f"{tracker_type} Tracking", frame, False)

    # Initialize tracker
    tracker = create_tracker(tracker_type)
    if tracker is None:
        return

    ret = tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        timer = cv2.getTickCount()

        # Update tracker
        success, bbox = tracker.update(frame)

        # Calculate FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if success:
            # Draw bounding box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), 2)

            # Show tracking status
            cv2.putText(frame, "Tracking", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show tracker type
        cv2.putText(frame, f"{tracker_type} Tracker", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display frame
        cv2.imshow(f"{tracker_type} Tracking", frame)

        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Try trackers in order of performance (best to worst)
    trackers = ['CSRT', 'KCF', 'GOTURN']

    for tracker_type in trackers:
        print(f"\nTesting {tracker_type} tracker...")
        track_object(tracker_type=tracker_type)

        # Ask if user wants to try next tracker
        if tracker_type != trackers[-1]:
            response = input("Try next tracker? (y/n): ")
            if response.lower() != 'y':
                break