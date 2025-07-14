import cv2
import os

def draw_boxes(frame, detections, ids=None, color=(0,255,0)):
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        label = f"ID {ids[idx]}" if ids is not None and idx < len(ids) else "Player"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)