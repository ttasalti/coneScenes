"""
End-to-end pipeline:
1) Cluster LiDAR points into objects per scene.
2) Flag cone-like clusters via simple rules.
3) Attach odometry to each cluster using metadata.json.
4) Project clusters to global coordinates.
5) Rotate everything into Scene-1 local frame.
6) Plot global map and Scene-1 local map for quick checks.
"""

from clustering import process_directory
from cone_detector import detect_cones
from attach_odom import attach_odometry
from globalize_cones import add_global_coordinates
from rotate_to_scene1 import rotate_to_scene1
from plot import plot_points
from plot_overlay_scene1 import plot_overlay
import pandas as pd

if __name__ == "__main__":
    POINTS_DIR = "cfs_vargarda8/points"
    CLUSTER_CSV = "cluster_summary_all.csv"
    EPS = 0.3
    MIN_SAMPLES = 5
    SCALE = True

    process_directory(points_dir=POINTS_DIR, output_csv=CLUSTER_CSV, eps=EPS, min_samples=MIN_SAMPLES, scale=SCALE)

    OUTPUT_CSV = "cluster_summary_with_flags.csv"
    detect_cones(CLUSTER_CSV, OUTPUT_CSV)
    attach_odometry("cluster_summary_with_flags.csv", "cfs_vargarda8/metadata.json", "cones_with_odom.csv")

    df = pd.read_csv("cones_with_odom.csv")
    add_global_coordinates(df).to_csv("cones_global.csv", index=False)

    rotate_to_scene1("cones_global.csv", "cones_rotated_scene1.csv")

    plot_points("cones_global.csv", "x_global", "y_global", "global_cones.png", "Global Cone Map")
    plot_points("cones_rotated_scene1.csv", "x_scene1", "y_scene1", "scene1_cones.png", "Scene-1 Local Cone Map")
    plot_overlay("cfs_vargarda8/labels/0000001.txt", "cones_rotated_scene1.csv", "overlay_scene1.png")
