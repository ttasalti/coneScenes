"""
Overlays Scene-1 label cones (local) with rotated global cones on the same 2D plot.
Useful to visually verify alignment between labels and computed positions.
"""

import pandas as pd
import matplotlib.pyplot as plt

def _load_labels_txt(path: str) -> pd.DataFrame:
    cols = ['x','y','z','dx','dy','dz','yaw','class']
    return pd.read_csv(path, sep=' ', names=cols)

def plot_overlay(labels_txt: str = "cfs_vargarda8/labels/0000001.txt",
                 rotated_csv: str = "cones_rotated_scene1.csv",
                 out_png: str = "overlay_scene1.png") -> None:
    lab = _load_labels_txt(labels_txt)
    cones = pd.read_csv(rotated_csv)
    plt.figure(figsize=(8,8))
    plt.scatter(lab['x'], lab['y'], s=10, label='labels_scene1')
    plt.scatter(cones['x_scene1'], cones['y_scene1'], s=10, label='rotated_global')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    plot_overlay("cfs_vargarda8/labels/0000001.txt",
                 "cones_rotated_scene1.csv",
                 "overlay_scene1.png")