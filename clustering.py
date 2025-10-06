"""
Loads LiDAR point clouds (*.bin), clusters points with DBSCAN on (x,y,z),
and writes per-cluster summaries (size, ranges, means, intensity stats) to CSV.
This is the first step: producing cluster candidates scene-by-scene.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_bin_points(path: Path) -> pd.DataFrame:
    data = np.fromfile(path, dtype=np.float32)
    arr = data.reshape(-1, 4)
    return pd.DataFrame(arr, columns=['x', 'y', 'z', 'intensity'])

def summarize_clusters(df: pd.DataFrame, scene: str, eps: float, min_samples: int, scale: bool) -> pd.DataFrame:
    feat = df[['x', 'y']].dropna().copy()
    X = feat.values
    if scale:
        X = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    out = df.copy()
    out['cluster'] = -1
    out.loc[feat.index, 'cluster'] = labels
    valid = out['cluster'] >= 0
    if valid.sum() == 0:
        cols = ['scene','cluster','N_points','dx','dy','dz','x_mean','y_mean','z_mean','intensity_mean','intensity_std']
        return pd.DataFrame(columns=cols)
    g = out[valid].groupby('cluster')
    r = g.agg(
        x_min=('x','min'), x_max=('x','max'),
        y_min=('y','min'), y_max=('y','max'),
        z_min=('z','min'), z_max=('z','max'),
        x_mean=('x','mean'), y_mean=('y','mean'), z_mean=('z','mean'),
        intensity_mean=('intensity','mean'), intensity_std=('intensity','std'),
        N_points=('x','size')
    ).reset_index()
    r['dx'] = r['x_max'] - r['x_min']
    r['dy'] = r['y_max'] - r['y_min']
    r['dz'] = r['z_max'] - r['z_min']
    r = r[['cluster','N_points','dx','dy','dz','x_mean','y_mean','z_mean','intensity_mean','intensity_std']]
    r.insert(0, 'scene', scene)
    return r

def process_directory(points_dir: str,
                      output_csv: str = "cluster_summary_all.csv",
                      eps: float = 0.5,
                      min_samples: int = 10,
                      scale: bool = True) -> None:
    p = Path(points_dir)
    if Path(output_csv).exists():
        Path(output_csv).unlink()
    wrote = False
    for bin_path in sorted(p.glob("*.bin")):
        scene = bin_path.stem
        df = load_bin_points(bin_path)
        summary = summarize_clusters(df, scene, eps=eps, min_samples=min_samples, scale=scale)
        if len(summary) > 0:
            summary.to_csv(output_csv, mode='a', index=False, header=not wrote)
            wrote = True
