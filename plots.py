"""
Figures for the README: one scan with its detections, recall and error by
range, and the cone map against the labelled track.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CONE_COLOURS = {'Cone_Yellow': '#E0A800', 'Cone_Blue': '#1F5FBF'}
MISS = '#C0392B'
INK = '#222222'

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def plot_scan(points, labels, visible, cones, matched_det, matched_gt, out_png, title=''):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c='#B5B5B5', linewidths=0, label='LiDAR points (ground removed)')
    vis = labels[visible]
    for cls, colour in CONE_COLOURS.items():
        sel = vis['class'] == cls
        ax.scatter(vis.loc[sel, 'x'], vis.loc[sel, 'y'], s=70, facecolors='none', edgecolors=colour,
                   linewidths=1.6, label=f'labelled {cls.split("_")[1].lower()} cone')
    missed = np.setdiff1d(np.arange(len(vis)), matched_gt)
    ax.scatter(vis['x'].values[missed], vis['y'].values[missed], s=150, marker='s', facecolors='none',
               edgecolors=MISS, linewidths=1.4, label='missed cone')
    hit = np.zeros(len(cones), dtype=bool)
    hit[matched_det] = True
    ax.scatter(cones['x'][hit], cones['y'][hit], s=22, marker='+', c=INK, linewidths=1.2, label='detection')
    ax.scatter(cones['x'][~hit], cones['y'][~hit], s=28, marker='x', c=MISS, linewidths=1.4,
               label='detection, no visible cone')
    ax.scatter([0], [0], marker='>', s=90, c=INK)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m), forward')
    ax.set_ylabel('y (m), left')
    ax.set_title(title, loc='left')
    ax.legend(loc='upper left', fontsize=8, frameon=False, markerscale=1.0)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_by_range(by_range, out_png):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(by_range))
    names = by_range['range_bin'].astype(str)
    a1.bar(x - 0.2, by_range['recall'], 0.4, color='#1F5FBF', label='recall')
    a1.bar(x + 0.2, by_range['precision'], 0.4, color='#9DB7E0', label='precision')
    for i, (r, p) in enumerate(zip(by_range['recall'], by_range['precision'])):
        a1.text(i - 0.2, r + 0.01, f'{r:.2f}', ha='center', fontsize=8)
        a1.text(i + 0.2, p + 0.01, f'{p:.2f}', ha='center', fontsize=8)
    a1.set_xticks(x, names)
    a1.set_ylim(0, 1.1)
    a1.set_xlabel('distance from the car (m)')
    a1.legend(frameon=False, fontsize=8, loc='lower left')
    a1.set_title('Detection, held-out scans', loc='left')
    a2.bar(x, by_range['error_mean'] * 100, 0.5, color='#1F5FBF')
    for i, e in enumerate(by_range['error_mean'] * 100):
        a2.text(i, e + 0.3, f'{e:.1f}', ha='center', fontsize=8)
    a2.set_xticks(x, names)
    a2.set_xlabel('distance from the car (m)')
    a2.set_ylabel('cm')
    a2.set_title('Mean position error of matched cones', loc='left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_map(track, landmarks, trajectory, out_png, title=''):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cls, colour in CONE_COLOURS.items():
        sel = track['class'] == cls
        ax.scatter(track.loc[sel, 'x'], track.loc[sel, 'y'], s=70, facecolors='none', edgecolors=colour,
                   linewidths=1.6, label=f'labelled {cls.split("_")[1].lower()} cone')
    ax.plot(trajectory[:, 0], trajectory[:, 1], c='#999999', lw=0.8, ls=':', marker='o', ms=2.5, label='car position at each scan')
    ax.scatter(landmarks['x'], landmarks['y'], s=22, marker='+', c=INK, linewidths=1.2, label='mapped cone')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, loc='left')
    ax.legend(loc='best', fontsize=8, frameon=False)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
