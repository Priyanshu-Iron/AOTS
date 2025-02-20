# Save as check_opencv.py
import cv2
print("OpenCV version:", cv2.__version__)
print("GOTURN available:", hasattr(cv2, 'TrackerGOTURN_create'))
print("CSRT available:", hasattr(cv2, 'TrackerCSRT_create'))
print("KCF available:", hasattr(cv2, 'TrackerKCF_create'))
with open('opencv_build_info.txt', 'w') as f:
    f.write(cv2.getBuildInformation())