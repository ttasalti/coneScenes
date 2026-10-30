"""
Reads a coneScenes scene: LiDAR scans, cone labels and odometry.

Each scan is a float32 array of (x, y, z, intensity) in the LiDAR frame. Each
label file lists every cone of the track in the frame of that scan, also the
cones the sensor cannot see. The odometry in metadata.json is stored as the
map origin seen from the car, so a point p in the scan frame maps to the world
as R(-yaw) (p - (x, y)).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLUMNS = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'yaw', 'class']


def load_scene(scene_dir: str) -> list:
    root = Path(scene_dir)
    with open(root / 'metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    scans = []
    for r in sorted(meta['data'], key=lambda r: r['id']):
        scans.append({
            'id': r['id'],
            'points': root / r['pointcloud']['file'],
            'labels': root / r['labels']['file'],
            'odom': r['odom'],
        })
    return scans


def load_points(path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_labels(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=' ', names=LABEL_COLUMNS)


def scan_to_world(xy: np.ndarray, odom: dict) -> np.ndarray:
    c, s = np.cos(odom['yaw']), np.sin(odom['yaw'])
    dx = xy[:, 0] - odom['x']
    dy = xy[:, 1] - odom['y']
    return np.c_[c * dx + s * dy, -s * dx + c * dy]
