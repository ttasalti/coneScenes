"""
End-to-end pipeline:
1) Remove the ground and the points near the car in every scan.
2) Tune DBSCAN and the cone rules on the first half of the scans.
3) Score detection on the second half, overall and by distance, next to the
   first version of the pipeline.
4) Move the detections of all scans to the world frame and build a cone map.
5) Write the tables to results/ and the figures to figures/.

The first version of the pipeline clusters the full scans and takes about
half an hour for the scene, so it is only scored with --with-first-version.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import baseline_cones
from clustering import cluster_scan
from cone_detector import detect_cones
from dataset import load_labels, load_points, load_scene, scan_to_world
from evaluate import evaluate_scan, match, summarise, visible_mask
from ground import remove_ground
from mapping import build_map, score_map, track_in_world
from plots import plot_by_range, plot_map, plot_scan
from tune import grid_search, split_scans

SCENE_DIR = 'cfs_vargarda8'
RANGE_BINS = [0, 10, 20, 30]
EXAMPLE_SCAN = 40
N_JOBS = int(os.environ.get('N_JOBS', 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-first-version', action='store_true')
    args = parser.parse_args()
    Path('results').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)

    prepared = []
    for s in load_scene(SCENE_DIR):
        raw = load_points(s['points'])
        points = remove_ground(raw)
        labels = load_labels(s['labels'])
        prepared.append({**s, 'raw': raw, 'points': points, 'labels': labels,
                         'visible': visible_mask(labels, points)})
    tune_scans, test_scans = split_scans(prepared)

    params, results = grid_search(tune_scans, n_jobs=N_JOBS)
    results.head(20).to_csv('results/tuning_top20.csv', index=False)
    with open('results/params.json', 'w') as f:
        json.dump(params, f, indent=2)

    def run(p):
        return detect_cones(cluster_scan(p['points'], params['eps'], params['min_samples']), params['rules'])

    rows, base_rows, outcomes = [], [], {}
    for name, scans in [('tuning scans', tune_scans), ('held-out scans', test_scans)]:
        outcomes[name] = pd.concat([evaluate_scan(run(p), p['labels'], p['visible']) for p in scans])
        rows.append({'split': name, 'scans': len(scans), **summarise(outcomes[name])})
        if args.with_first_version:
            base = pd.concat([evaluate_scan(baseline_cones(p['raw']), p['labels'], p['visible']) for p in scans])
            base_rows.append({'split': name, 'scans': len(scans), **summarise(base)})
    pd.DataFrame(rows).to_csv('results/detection.csv', index=False)
    if base_rows:
        pd.DataFrame(base_rows).to_csv('results/detection_first_version.csv', index=False)

    test = outcomes['held-out scans']
    test['range_bin'] = pd.cut(test['range'], RANGE_BINS)
    by_range = pd.DataFrame([{'range_bin': f'{b.left}-{b.right}', **summarise(g)}
                             for b, g in test.groupby('range_bin', observed=False)])
    by_range.to_csv('results/detection_by_range.csv', index=False)
    plot_by_range(by_range, 'figures/detection_by_range.png')

    ex = next(p for p in prepared if p['id'] == EXAMPLE_SCAN)
    cones = run(ex).reset_index(drop=True)
    vis = ex['labels'][ex['visible']].reset_index(drop=True)
    di, gi, _ = match(cones[['x', 'y']].values, vis[['x', 'y']].values)
    plot_scan(ex['points'], ex['labels'], ex['visible'], cones, di, gi, 'figures/scan_example.png',
              f'Scan {EXAMPLE_SCAN} (held out): {len(gi)} of {len(vis)} visible cones found')

    detections = []
    for p in prepared:
        c = run(p)
        xy = scan_to_world(c[['x', 'y']].values, p['odom'])
        detections.append(pd.DataFrame({'scan': p['id'], 'x_world': xy[:, 0], 'y_world': xy[:, 1]}))
    landmarks = build_map(pd.concat(detections, ignore_index=True))
    track = track_in_world(prepared[0]['labels'], prepared[0]['odom'])
    seen = np.zeros(len(track), dtype=bool)
    for p in prepared:
        seen |= p['visible']
    map_rows = [{'reference': 'all labelled cones', **score_map(landmarks, track)},
                {'reference': 'cones visible in at least one scan', **score_map(landmarks, track[seen])}]
    pd.DataFrame(map_rows).to_csv('results/map.csv', index=False)
    trajectory = np.vstack([scan_to_world(np.zeros((1, 2)), p['odom']) for p in prepared])
    plot_map(track, landmarks, trajectory, 'figures/cone_map.png',
             f'Cone map from {len(prepared)} scans: {map_rows[0]["matched"]} of {len(track)} track cones mapped')
