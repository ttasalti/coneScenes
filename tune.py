"""
Grid search over the DBSCAN parameters and the cone rules. The scene is split
in time: the first half of the scans is used for tuning, the second half is
kept for the reported scores. Parameters are chosen by F1 on the tuning half;
the centre offset only moves detections, so it is chosen afterwards by the
mean localisation error.
"""

import itertools

import pandas as pd
from joblib import Parallel, delayed

from clustering import cluster_scan
from cone_detector import DEFAULT_RULES, detect_cones
from evaluate import evaluate_scan, summarise

DBSCAN_GRID = {'eps': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 'min_samples': [2, 3, 4]}
RULE_GRID = {
    'max_extent': [0.3, 0.4, 0.5, 0.6, 0.7],
    'min_height': [0.1, 0.15, 0.2],
    'max_height': [0.4, 0.45, 0.5, 0.6],
    'min_points': [3, 5, 8],
}
OFFSET_GRID = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]


def split_scans(prepared: list) -> tuple:
    half = len(prepared) // 2
    return prepared[:half], prepared[half:]


def score(prepared: list, clusters: list, rules: dict) -> dict:
    outcomes = [evaluate_scan(detect_cones(c, rules), p['labels'], p['visible']) for p, c in zip(prepared, clusters)]
    return summarise(pd.concat(outcomes))


def _search_rules(prepared: list, eps: float, min_samples: int) -> list:
    clusters = [cluster_scan(p['points'], eps, min_samples) for p in prepared]
    rows = []
    for values in itertools.product(*RULE_GRID.values()):
        rules = dict(DEFAULT_RULES, **dict(zip(RULE_GRID, values)))
        rows.append({'eps': eps, 'min_samples': min_samples, **rules, **score(prepared, clusters, rules)})
    return rows


def grid_search(prepared: list, n_jobs: int = 1) -> tuple:
    prepared = [{k: p[k] for k in ('points', 'labels', 'visible')} for p in prepared]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_search_rules)(prepared, eps, min_samples)
        for eps, min_samples in itertools.product(*DBSCAN_GRID.values())
    )
    results = pd.DataFrame([r for part in parts for r in part]).sort_values('f1', ascending=False)
    best = results.iloc[0]
    rules = {k: best[k] for k in DEFAULT_RULES}
    rules['min_points'] = int(rules['min_points'])
    clusters = [cluster_scan(p['points'], best['eps'], int(best['min_samples'])) for p in prepared]
    offsets = {o: score(prepared, clusters, dict(rules, centre_offset=o))['error_mean'] for o in OFFSET_GRID}
    rules['centre_offset'] = min(offsets, key=offsets.get)
    params = {'eps': float(best['eps']), 'min_samples': int(best['min_samples']), 'rules': rules}
    return params, results
