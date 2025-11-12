\
import math
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Colors are BGR for OpenCV
COLORS = {
    "pose": (40, 180, 255),
    "gaze": (50, 220, 50),
    "bbox": (220, 120, 60),
    "text": (255, 255, 255),
}
def draw_bbox(img: np.ndarray, bbox, color=(0, 255, 0), thickness=2):
    # Handle bounding boxes of variable length safely
    if bbox is None:
        return
    if len(bbox) > 4:
        bbox = bbox[:4]  # Use only first four values
    x, y, w, h = map(float, bbox)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

def yaw_pitch_to_vector(yaw: float, pitch: float, length: float = 50.0) -> Tuple[float, float]:
    # yaw: left(-) to right(+), pitch: down(+) to up(-) using a simple convention
    dx = length * math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
    dy = -length * math.sin(math.radians(pitch))
    return dx, dy

def draw_head_pose(img: np.ndarray, center: Tuple[int, int], yaw: float, pitch: float, color=None, thickness=2):
    if color is None:
        color = COLORS["pose"]
    cx, cy = int(center[0]), int(center[1])
    dx, dy = yaw_pitch_to_vector(yaw, pitch, length=60.0)
    cv2.arrowedLine(img, (cx, cy), (int(cx + dx), int(cy + dy)), color, thickness, tipLength=0.25)
    cv2.circle(img, (cx, cy), 3, color, -1)

def put_text(img: np.ndarray, text: str, org=(10, 20), color=None, scale=0.6, thickness=1):
    if color is None:
        color = COLORS["text"]
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def visualize_predictions(
    img: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    yaw: float,
    pitch: float,
    gaze_label: Optional[int],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    cx, cy = w // 2, h // 2
    if bbox is not None:
        draw_bbox(vis, bbox, COLORS["bbox"], 2)
        cx = int(bbox[0] + bbox[2] / 2)
        cy = int(bbox[1] + bbox[3] / 2)
    draw_head_pose(vis, (cx, cy), yaw, pitch, COLORS["pose"], 2)
    label = f"gaze={gaze_label}"
    if class_names and gaze_label is not None and 0 <= gaze_label < len(class_names):
        label = f"gaze={class_names[gaze_label]}"
    put_text(vis, f"yaw={yaw:.1f}, pitch={pitch:.1f}", (10, 20))
    put_text(vis, label, (10, 45))
    return vis

def save_image(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
