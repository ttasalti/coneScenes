"""
Takes cluster summaries and applies simple rule-based thresholds on size and
point count to mark clusters as cones (is_cone=True/False). Writes only the
rows where is_cone=True into a new CSV file.
"""

import pandas as pd

def detect_cones(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    if 'dx' not in df.columns or 'dy' not in df.columns or 'dz' not in df.columns or 'N_points' not in df.columns:
        df.to_csv(output_csv, index=False)
        return
    dx_ok = (df['dx'] >= 0.05) & (df['dx'] <= 0.23)
    dy_ok = (df['dy'] >= 0.05) & (df['dy'] <= 0.23)
    dz_ok = (df['dz'] >= 0.10) & (df['dz'] <= 0.33)
    n_ok  = (df['N_points'] >= 5) & (df['N_points'] <= 200)
    df['is_cone'] = dx_ok & dy_ok & dz_ok & n_ok
    df[df['is_cone'] == True].to_csv(output_csv, index=False)
