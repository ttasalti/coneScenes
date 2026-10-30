"""
Removes the points close to the car and the ground. The ground is a plane
z = a*x + b*y + c fitted with RANSAC and refined by least squares on its
inliers; every remaining point gets its height above that plane.
"""

import numpy as np


def crop(points: np.ndarray, min_range: float = 1.5, max_range: float = 30.0) -> np.ndarray:
    r = np.hypot(points[:, 0], points[:, 1])
    return points[(r >= min_range) & (r <= max_range)]


def fit_ground_plane(points: np.ndarray, threshold: float = 0.04, iterations: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = points[:, :3].astype(np.float64)
    best, best_count = None, -1
    for _ in range(iterations):
        p = xyz[rng.choice(len(xyz), 3, replace=False)]
        A = np.c_[p[:, 0], p[:, 1], np.ones(3)]
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        plane = np.linalg.solve(A, p[:, 2])
        count = (np.abs(xyz[:, 2] - xyz[:, :2] @ plane[:2] - plane[2]) < threshold).sum()
        if count > best_count:
            best, best_count = plane, count
    inliers = np.abs(xyz[:, 2] - xyz[:, :2] @ best[:2] - best[2]) < threshold
    A = np.c_[xyz[inliers, 0], xyz[inliers, 1], np.ones(inliers.sum())]
    plane, *_ = np.linalg.lstsq(A, xyz[inliers, 2], rcond=None)
    return plane


def remove_ground(points: np.ndarray, min_height: float = 0.05, max_height: float = 0.6, **kwargs) -> np.ndarray:
    """Returns the non-ground points as (x, y, height above ground, intensity)."""
    pts = crop(points, **kwargs)
    plane = fit_ground_plane(pts)
    h = pts[:, 2] - pts[:, :2] @ plane[:2] - plane[2]
    keep = (h >= min_height) & (h <= max_height)
    return np.c_[pts[keep, 0], pts[keep, 1], h[keep], pts[keep, 3]]
