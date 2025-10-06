"""
Reads metadata.json and attaches odometry fields (x,y,z,yaw,vx,vy,yawrate,timestamp)
to each cluster row by scene key matching. Produces a CSV with odometry columns.
"""

import json
import pandas as pd
from pathlib import Path

def _scene_key_from_path(s: str) -> str:
    name = Path(s).stem
    if name.isdigit():
        return f"{int(name):07d}"
    return name

def _normalize_scene(s) -> str:
    s = str(s).strip()
    if s.isdigit():
        return f"{int(s):07d}"
    return s

def attach_odometry(input_csv: str, metadata_json: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    with open(metadata_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    rows = meta.get('data', [])
    m = {}
    for r in rows:
        od = r.get('odom', {})
        pc = r.get('pointcloud', {}).get('file', '')
        lb = r.get('labels', {}).get('file', '')
        keys = set()
        if pc:
            keys.add(_scene_key_from_path(pc))
        if lb:
            keys.add(_scene_key_from_path(lb))
        if 'id' in r:
            keys.add(f"{int(r['id']):07d}")
        for k in keys:
            m[k] = {
                'odom_timestamp': od.get('timestamp', None),
                'odom_x': od.get('x', None),
                'odom_y': od.get('y', None),
                'odom_z': od.get('z', None),
                'odom_yaw': od.get('yaw', None),
                'odom_vx': od.get('vx', None),
                'odom_vy': od.get('vy', None),
                'odom_yawrate': od.get('yawrate', None),
            }
    if 'scene' in df.columns:
        df['scene_key'] = df['scene'].map(_normalize_scene)
    else:
        df['scene_key'] = None
    od_df = pd.DataFrame.from_dict(m, orient='index').reset_index().rename(columns={'index':'scene_key'})
    out = pd.merge(df, od_df, on='scene_key', how='left')
    out.drop(columns=['scene_key'], inplace=True)
    out.to_csv(output_csv, index=False)
