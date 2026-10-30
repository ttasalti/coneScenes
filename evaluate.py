"""
Scores detections against the labels of the same scan.

A label file lists the whole track, so recall is computed over the visible
cones only: labels with at least `min_visible_points` non-ground LiDAR points
within `visible_radius` of their centre. Detections and visible labels are
matched one to one (Hungarian algorithm) within `gate` metres. A detection that
lands on a labelled but not visible cone is neither a true nor a false positive.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def visible_mask(labels: pd.DataFrame, points: np.ndarray, visible_radius: float = 0.3,
                 min_visible_points: int = 3, max_range: float = 30.0) -> np.ndarray:
    xy = labels[['x', 'y']].values
    if len(points) == 0:
        return np.zeros(len(xy), dtype=bool)
    counts = np.array([len(i) for i in cKDTree(points[:, :2]).query_ball_point(xy, visible_radius)])
    return (counts >= min_visible_points) & (np.hypot(xy[:, 0], xy[:, 1]) <= max_range)


def match(det_xy: np.ndarray, gt_xy: np.ndarray, gate: float = 0.5):
    """Returns matched (det index, gt index) pairs and their distances."""
    if len(det_xy) == 0 or len(gt_xy) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    d = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(np.where(d <= gate, d, 1e6))
    ok = d[rows, cols] <= gate
    return rows[ok], cols[ok], d[rows[ok], cols[ok]]


def evaluate_scan(cones: pd.DataFrame, labels: pd.DataFrame, visible: np.ndarray, gate: float = 0.5) -> pd.DataFrame:
    """One row per visible label and per unmatched detection, with its outcome."""
    det_xy = cones[['x', 'y']].values
    all_xy = labels[['x', 'y']].values
    vis_idx = np.flatnonzero(visible)
    di, gi, dist = match(det_xy, all_xy[vis_idx], gate)
    rows = []
    for k, j in enumerate(vis_idx):
        hit = np.flatnonzero(gi == k)
        rows.append({
            'kind': 'tp' if len(hit) else 'fn',
            'range': float(np.hypot(*all_xy[j])),
            'error': float(dist[hit[0]]) if len(hit) else np.nan,
            'class': labels['class'].iloc[j],
        })
    unmatched = np.setdiff1d(np.arange(len(det_xy)), di)
    if len(unmatched):
        # detections that sit on a hidden labelled cone are not counted as false
        hidden = all_xy[~visible]
        near_hidden = np.zeros(len(unmatched), dtype=bool)
        if len(hidden):
            near_hidden = cKDTree(hidden).query(det_xy[unmatched])[0] <= gate
        for i, h in zip(unmatched, near_hidden):
            if not h:
                rows.append({'kind': 'fp', 'range': float(np.hypot(*det_xy[i])), 'error': np.nan, 'class': None})
    return pd.DataFrame(rows, columns=['kind', 'range', 'error', 'class'])


def summarise(outcomes: pd.DataFrame) -> dict:
    tp = (outcomes['kind'] == 'tp').sum()
    fn = (outcomes['kind'] == 'fn').sum()
    fp = (outcomes['kind'] == 'fp').sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    err = outcomes.loc[outcomes['kind'] == 'tp', 'error']
    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'precision': precision, 'recall': recall, 'f1': f1,
        'error_mean': float(err.mean()) if len(err) else np.nan,
        'error_median': float(err.median()) if len(err) else np.nan,
    }
