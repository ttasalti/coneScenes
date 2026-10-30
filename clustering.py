"""
Clusters the non-ground points of a scan with DBSCAN on (x, y) in metres and
summarises each cluster: point count, footprint, height and centroid.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

SUMMARY_COLUMNS = ['cluster', 'n_points', 'x', 'y', 'dx', 'dy', 'height', 'range', 'intensity_mean']


def cluster_scan(points: np.ndarray, eps: float = 0.3, min_samples: int = 3) -> pd.DataFrame:
    if len(points) == 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points[:, :2])
    df = pd.DataFrame(points, columns=['x', 'y', 'h', 'intensity'])
    df['cluster'] = labels
    g = df[df['cluster'] >= 0].groupby('cluster')
    r = g.agg(
        n_points=('x', 'size'),
        x=('x', 'mean'), y=('y', 'mean'),
        x_min=('x', 'min'), x_max=('x', 'max'),
        y_min=('y', 'min'), y_max=('y', 'max'),
        height=('h', 'max'),
        intensity_mean=('intensity', 'mean'),
    ).reset_index()
    r['dx'] = r['x_max'] - r['x_min']
    r['dy'] = r['y_max'] - r['y_min']
    r['range'] = np.hypot(r['x'], r['y'])
    return r[SUMMARY_COLUMNS]
