\
import numpy as np
from typing import Dict, Tuple

def compose_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return 3x4 projection matrix P = K [R|t]."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt

def project_points(points_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project Nx3 3D points to Nx2 image points using intrinsics K and extrinsics (R, t)."""
    P = compose_projection(K, R, t)  # 3x4
    homog = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # Nx4
    proj = (P @ homog.T).T  # Nx3
    proj = proj[:, :2] / proj[:, 2:3].clip(min=1e-6)
    return proj

def triangulate_points(pts1: np.ndarray, P1: np.ndarray, pts2: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Linear triangulation for corresponding 2D points from two views.
    pts: Nx2, P: 3x4 -> returns Nx3"""
    n = pts1.shape[0]
    X = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.array([
            x1*P1[2]-P1[0],
            y1*P1[2]-P1[1],
            x2*P2[2]-P2[0],
            y2*P2[2]-P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X[i] = (X_h[:3] / (X_h[3] + 1e-8))
    return X
