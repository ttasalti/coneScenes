"""
Builds a cone map of the track. Detections from every scan are moved to the
world frame with the odometry, grouped with DBSCAN, and every group seen in at
least `min_scans` scans becomes one landmark at the mean position. The map is
scored against the labels of the first scan moved to the world frame.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from dataset import scan_to_world
from evaluate import match


def build_map(detections: pd.DataFrame, eps: float = 0.5, min_scans: int = 2) -> pd.DataFrame:
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(detections[['x_world', 'y_world']].values)
    d = detections.assign(landmark=labels)
    m = d.groupby('landmark').agg(
        x=('x_world', 'mean'), y=('y_world', 'mean'), n_scans=('scan', 'nunique')
    ).reset_index()
    return m[m['n_scans'] >= min_scans].reset_index(drop=True)


def track_in_world(labels: pd.DataFrame, odom: dict) -> pd.DataFrame:
    xy = scan_to_world(labels[['x', 'y']].values, odom)
    return pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1], 'class': labels['class'].values})


def score_map(landmarks: pd.DataFrame, track: pd.DataFrame, gate: float = 0.5) -> dict:
    di, gi, dist = match(landmarks[['x', 'y']].values, track[['x', 'y']].values, gate)
    return {
        'landmarks': len(landmarks), 'track_cones': len(track), 'matched': len(di),
        'precision': len(di) / len(landmarks), 'recall': len(gi) / len(track),
        'error_mean': float(dist.mean()), 'error_median': float(np.median(dist)),
    }
