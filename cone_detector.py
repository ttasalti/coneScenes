"""
Keeps the clusters that look like cones. The labelled cones are 0.23 m wide
and 0.33 m tall, so a cluster must have a small footprint, a height close to
that of a cone and enough points. The LiDAR only sees the near face of a cone,
so the centroid is pushed back along the viewing ray by `centre_offset` metres.
"""

import numpy as np
import pandas as pd

DEFAULT_RULES = {
    'max_extent': 0.4,
    'min_height': 0.15,
    'max_height': 0.45,
    'min_points': 3,
    'centre_offset': 0.05,
}


def detect_cones(clusters: pd.DataFrame, rules: dict = DEFAULT_RULES) -> pd.DataFrame:
    extent = np.maximum(clusters['dx'], clusters['dy'])
    ok = (
        (extent <= rules['max_extent'])
        & (clusters['height'] >= rules['min_height'])
        & (clusters['height'] <= rules['max_height'])
        & (clusters['n_points'] >= rules['min_points'])
    )
    cones = clusters[ok].copy()
    scale = 1 + rules['centre_offset'] / cones['range']
    cones['x'] = cones['x'] * scale
    cones['y'] = cones['y'] * scale
    return cones
