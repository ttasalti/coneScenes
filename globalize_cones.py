"""
Projects cluster centroids from the vehicle local frame to global/world frame
using attached odometry (x,y,yaw). Adds x_global and y_global columns.
"""

import numpy as np
import pandas as pd

def add_global_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    x = df['x_mean']
    y = df['y_mean']
    ox = df['odom_x']
    oy = df['odom_y']
    yaw = df['odom_yaw']
    c = np.cos(yaw)
    s = np.sin(yaw)
    xg = ox + c * x - s * y
    yg = oy + s * x + c * y
    out = df.copy()
    out['x_global'] = xg
    out['y_global'] = yg
    return out
