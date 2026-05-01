import math
import numpy as np
from scipy import signal

def bbox_to_center(b):
    """Convert bbox [x1,y1,x2,y2] or dict {x1,y1,x2,y2} to center + size (cx,cy,w,h).
    
    - Validates bbox format and values
    - Ensures positive width/height
    - Handles both list/tuple and dict formats
    """
    if b is None:
        raise ValueError(f'Invalid bbox: {b}')
    
    # Handle both dict and list/tuple formats
    if isinstance(b, dict):
        x1, y1, x2, y2 = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])
    elif isinstance(b, (list, tuple)) and len(b) >= 4:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    else:
        raise ValueError(f'Invalid bbox: {b}')
    
    # Ensure x1 < x2, y1 < y2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    return cx, cy, w, h


def center_to_bbox(cx, cy, w, h):
    """Convert center + size (cx,cy,w,h) to bbox [x1,y1,x2,y2]."""
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def center_distance(b1, b2):
    """Compute Euclidean distance between centers of two bboxes.
    
    Args:
        b1, b2: bboxes in [x1,y1,x2,y2] format
    
    Returns:
        float: distance between centers
    """
    if b1 is None or b2 is None:
        return float('nan')
    
    try:
        cx1, cy1, _, _ = bbox_to_center(b1)
        cx2, cy2, _, _ = bbox_to_center(b2)
        return math.hypot(cx1 - cx2, cy1 - cy2)
    except (ValueError, TypeError):
        return float('nan')


def bbox_iou(b1, b2):
    """Compute Intersection over Union (IoU) of two bboxes."""
    if b1 is None or b2 is None:
        return 0.0
    
    x1_min, y1_min, x1_max, y1_max = b1[0], b1[1], b1[2], b1[3]
    x2_min, y2_min, x2_max, y2_max = b2[0], b2[1], b2[2], b2[3]
    
    # Intersection
    x_left = max(x1_min, x2_min)
    x_right = min(x1_max, x2_max)
    y_top = max(y1_min, y2_min)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def smooth_trajectory(trajectory, window=5, method='savgol'):
    """Smooth trajectory coordinates using moving average or Savitzky-Golay filter.
    
    Args:
        trajectory: array of values [x1, x2, ..., xn]
        window: window size for smoothing (must be odd for savgol)
        method: 'moving_avg' or 'savgol'
    
    Returns:
        smoothed trajectory (same length as input)
    """
    if len(trajectory) < window:
        return trajectory
    
    trajectory = np.array(trajectory, dtype=float)
    
    if method == 'savgol':
        if window % 2 == 0:
            window += 1  # Ensure odd
        if window > len(trajectory):
            window = len(trajectory) // 2
            if window % 2 == 0:
                window += 1
        if window < 3:
            return trajectory
        try:
            return signal.savgol_filter(trajectory, window, 2)
        except:
            return trajectory
    else:  # moving average
        if window > len(trajectory):
            window = len(trajectory)
        kernel = np.ones(window) / window
        padded = np.pad(trajectory, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(trajectory)]


def compute_velocity(positions, dt=1.0):
    """Compute velocity from position trajectory.
    
    Args:
        positions: array of position values [p1, p2, ..., pn]
        dt: time step between frames (default 1.0)
    
    Returns:
        velocity array (same length as input, padded with first velocity)
    """
    positions = np.array(positions, dtype=float)
    if len(positions) < 2:
        return np.zeros_like(positions)
    
    velocities = np.diff(positions) / dt
    # Pad first velocity with second velocity for consistency
    velocities = np.concatenate([[velocities[0]], velocities])
    return velocities