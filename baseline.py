"""
The first version of this pipeline, kept for comparison: DBSCAN with eps=0.3
and min_samples=5 on standardised (x, y) of the raw scan, without removing the
ground or the points near the car, and the original size rules on the
cluster's z range.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def baseline_cones(raw_points: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(raw_points, columns=['x', 'y', 'z', 'intensity'])
    X = StandardScaler().fit_transform(df[['x', 'y']].values)
    df['cluster'] = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
    g = df[df['cluster'] >= 0].groupby('cluster')
    r = g.agg(n=('x', 'size'), x=('x', 'mean'), y=('y', 'mean'),
              x_min=('x', 'min'), x_max=('x', 'max'), y_min=('y', 'min'), y_max=('y', 'max'),
              z_min=('z', 'min'), z_max=('z', 'max'))
    dx, dy, dz = r['x_max'] - r['x_min'], r['y_max'] - r['y_min'], r['z_max'] - r['z_min']
    ok = (dx.between(0.05, 0.23) & dy.between(0.05, 0.23) & dz.between(0.10, 0.33) & r['n'].between(5, 200))
    return r.loc[ok, ['x', 'y']].reset_index(drop=True)
