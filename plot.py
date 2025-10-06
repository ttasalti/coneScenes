"""
Draws a simple 2D scatter plot from a CSV given X and Y column names.
Saves the figure to a PNG with equal aspect for quick visual inspection.
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_points(csv_path: str, x_col: str, y_col: str, out_png: str, title: str = "") -> None:
    df = pd.read_csv(csv_path)
    x = df[x_col]
    y = df[y_col]
    plt.figure(figsize=(8,8))
    plt.scatter(x, y, s=10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
