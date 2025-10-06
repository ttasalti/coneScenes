"""
Re-expresses global cone coordinates relative to Scene-1 local frame by using
Scene-1 odometry as reference (or earliest timestamp if Scene-1 not found).
Adds x_scene1 and y_scene1, and writes a rotated CSV.
"""

import numpy as np
import pandas as pd

def _normalize_scene(s) -> str:
    s = str(s).strip()
    if s.isdigit():
        return f"{int(s):07d}"
    return s

def rotate_to_scene1(csv_in: str = "cones_global.csv", csv_out: str = "cones_rotated_scene1.csv") -> None:
    df = pd.read_csv(csv_in)
    if 'scene' in df.columns:
        df['scene_key'] = df['scene'].map(_normalize_scene)
    else:
        df['scene_key'] = None
    if (df['scene_key'] == '0000001').any():
        ref = df.loc[df['scene_key'] == '0000001'].iloc[0]
    elif 'odom_timestamp' in df.columns and df['odom_timestamp'].notna().any():
        ref = df.loc[df['odom_timestamp'].idxmin()]
    else:
        ref = df.iloc[0]
    rx = ref['odom_x']
    ry = ref['odom_y']
    rYaw = ref['odom_yaw']
    c = np.cos(rYaw)
    s = np.sin(rYaw)
    dx = df['x_global'] - rx
    dy = df['y_global'] - ry
    x1 = c * dx + s * dy
    y1 = -s * dx + c * dy
    out = df.copy()
    out['x_scene1'] = x1
    out['y_scene1'] = y1
    out.drop(columns=['scene_key'], inplace=True)
    out.to_csv(csv_out, index=False)
