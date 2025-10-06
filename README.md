# Cone Detection and Localization Pipeline

This project processes Formula Student Driverless LiDAR scenes (from the coneScenes dataset) to detect traffic cones, attach vehicle odometry, transform them into global coordinates, and re-express them relative to a reference scene (Scene 1).  
It includes clustering, rule-based cone filtering, coordinate transformations, and visualization.

---

### Project Structure

```text
project_root/
│
├── cfs_vargarda8/
│   ├── points/                 # LiDAR .bin files per scene
│   ├── labels/                 # Ground-truth label .txt files
│   └── metadata.json           # Odometry and scene metadata
│
├── clustering.py               # DBSCAN-based clustering (x, y)
├── cone_detector.py            # Rule-based cone filtering
├── attach_odom.py              # Attach odometry to clusters
├── globalize_cones.py          # Local → global transformation
├── rotate_to_scene1.py         # Global → Scene-1 local rotation
├── plot.py                     # Simple scatter plot function
├── plot_overlay_scene1.py      # Overlay ground-truth vs computed
└── main.py                     # Full end-to-end pipeline
```

### Generated Outputs
```text
cluster_summary_all.csv        - Raw DBSCAN cluster summaries per scene  
cluster_summary_with_flags.csv - Same, with is_cone boolean flag  
cones_with_odom.csv            - Clusters merged with odometry  
cones_global.csv               - Cones in global/world coordinates  
cones_rotated_scene1.csv       - Global cones transformed to Scene 1 frame  
global_cones.png               - Visualization of global cone positions  
scene1_cones.png               - Visualization of Scene 1 cones  
overlay_scene1.png             - Overlay of ground-truth and rotated cones
```

### Personal Notes

- Hyperparameter choices were initially adapted from the following reference implementation:  
  https://github.com/sanchezfdezjavier/Self-driving-LiDAR-cone-detection/blob/main/src/lidar_nodes/src/clustering.py

- DBSCAN is applied only on **(x, y)** coordinates. The z-axis and intensity are not used in clustering decisions; they are included only in per-cluster summaries.

- The algorithm is not yet fully stabilized. Results are sensitive to DBSCAN parameters (`eps`, `min_samples`), rule-based thresholds (`dx`, `dy`, `dz`, `N_points`), and transformation settings.

- In the transformation step (global → Scene-1), the applied formulas were initially drafted with AI assistance. This part needs to be revisited and rewritten with deeper domain knowledge.

- The rule-based cone size thresholds were determined based on observations from `label_summary.ipynb`, where maximum values of labeled cones defined the upper bounds for `dx`, `dy`, and `dz`.  
  Lower limits and point count ranges were chosen heuristically to filter out noise and can be adjusted depending on sensor characteristics or clustering density.

#### Future Development

- A **loss function** should be created to reduce the difference between detected and real cones, and to increase the number of correct detections at the same time.  
  With this loss function, it will be possible to test and tune both the thresholds and DBSCAN hyperparameters.

- Instead of the current rule-based method, more advanced **cone detection techniques** from research papers can be studied and applied to make the system more reliable.

- By using **previous frames** together, time-series based approaches can be developed to achieve more stable and consistent cone detection results.

- A new step can be added to **rebuild the track** using the detected cones.

- After building the track, an **ideal racing line** can be found by using geometric or physical models.

- Later, **reinforcement learning** experiments can be started to learn better driving strategies using the reconstructed track and cone information.

