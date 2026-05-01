# ========================= filling.py =========================
import numpy as np
from scipy.interpolate import CubicSpline
from geometry import bbox_to_center, center_to_bbox


# ---------------- FIX SIZE ----------------
def fix_wh(w, h):
    w = max(1.0, min(float(w), 2000.0))
    h = max(1.0, min(float(h), 2000.0))
    return w, h


# ---------------- Kalman2D + RTS ----------------
class Kalman2D:
    def __init__(self, dt=1.0, q=1e-3, r=1e-2):
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r

    def smooth(self, fids, centers):
        """
        fids: array-like frame_id có measurement
        centers: array-like [(cx,cy), ...]
        return: dict {fid: (cx, cy)}  (CHỈ real_fids)
        """
        n = len(fids)
        X = np.zeros((n,4))
        P = np.zeros((n,4,4))

        X[0,:2] = centers[0]
        P[0] = np.eye(4)

        # Forward
        for i in range(1, n):
            X_pred = self.F @ X[i-1]
            P_pred = self.F @ P[i-1] @ self.F.T + self.Q

            z = np.array(centers[i])
            y = z - (self.H @ X_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            X[i] = X_pred + K @ y
            P[i] = (np.eye(4) - K @ self.H) @ P_pred

        # Backward RTS
        for i in range(n-2, -1, -1):
            P_pred = self.F @ P[i] @ self.F.T + self.Q
            C = P[i] @ self.F.T @ np.linalg.inv(P_pred)
            X[i] += C @ (X[i+1] - self.F @ X[i])

        return {int(fids[i]): (float(X[i,0]), float(X[i,1])) for i in range(n)}


# ---------------- MAIN FILL FUNCTION ----------------
def fill_bbox_track(track, frame_range):
    """
    track: dict {frame_id: bbox[x1,y1,x2,y2]}
    frame_range: (start_fid, end_fid)
    return: dict {frame_id: bbox[x1,y1,x2,y2]}  (FULL RANGE)
    """
    start_f, end_f = frame_range
    all_fids = list(range(start_f, end_f + 1))

    if not track:
        return {f: [0,0,0,0] for f in all_fids}

    # ---- collect real data ----
    real_fids, centers, widths, heights = [], [], [], []
    for fid, bbox in track.items():
        try:
            cx, cy, w, h = bbox_to_center(bbox)
            real_fids.append(fid)
            centers.append((cx, cy))
            widths.append(w)
            heights.append(h)
        except:
            continue

    if len(real_fids) < 2:
        return {f: track.get(f, [0,0,0,0]) for f in all_fids}

    real_fids = np.array(real_fids)
    centers = np.array(centers)
    widths = np.array(widths)
    heights = np.array(heights)

    # ---- Kalman smoothing (CENTER) ----
    kf = Kalman2D()
    smooth_centers_real = kf.smooth(real_fids, centers)

    # ---- interpolate CENTER to all frames ----
    real_sorted = np.array(sorted(smooth_centers_real.keys()))
    cx_vals = np.array([smooth_centers_real[f][0] for f in real_sorted])
    cy_vals = np.array([smooth_centers_real[f][1] for f in real_sorted])

    cx_interp = CubicSpline(real_sorted, cx_vals, bc_type="natural")
    cy_interp = CubicSpline(real_sorted, cy_vals, bc_type="natural")

    smooth_centers = {
        f: (float(cx_interp(f)), float(cy_interp(f)))
        for f in all_fids
    }

    # ---- spline WIDTH / HEIGHT ----
    w_interp = CubicSpline(real_fids, widths, bc_type="natural")
    h_interp = CubicSpline(real_fids, heights, bc_type="natural")

    # ---- build output ----
    filled = {}
    for f in all_fids:
        if f in track:
            filled[f] = track[f]
        else:
            cx, cy = smooth_centers[f]
            w, h = fix_wh(w_interp(f), h_interp(f))
            filled[f] = center_to_bbox(cx, cy, w, h)

    return filled
