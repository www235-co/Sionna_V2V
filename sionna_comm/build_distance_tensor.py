import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path

def build_vehicle_distance_tensor(root_dir):
    root_path = Path(root_dir)
    vehicle_ids = [d.name for d in root_path.iterdir() if d.is_dir()]
    vehicle_ids.sort()
    vehicle_id2idx = {vid: idx for idx, vid in enumerate(vehicle_ids)}
    vehicle_num = len(vehicle_ids) if vehicle_ids else 0
    all_frame_nums = []
    for vid in vehicle_ids:
        vid_path = root_path / vid
        for yaml_file in vid_path.glob("0*.yaml"): 
            frame_prefix = yaml_file.name.split(".yaml")[0]
            frame_num_str = ''.join([c for c in frame_prefix if c.isdigit()])
            if frame_num_str: 
                all_frame_nums.append(frame_num_str)

    unique_frame_nums = sorted(list(set(all_frame_nums)), key=lambda x: int(x))
    frame_num2idx = {fn: idx for idx, fn in enumerate(unique_frame_nums)}
    frame_num = len(unique_frame_nums) if unique_frame_nums else 0

    if frame_num == 0 or vehicle_num == 0:
        empty_tensor = tf.constant([], dtype=tf.float32, shape=[0, 0, 0])
        empty_mask = tf.constant([], dtype=tf.float32, shape=[0, 0, 0])
        return empty_tensor, empty_mask, frame_num2idx, vehicle_id2idx

    coord_dict = {v_idx: {} for v_idx in range(vehicle_num)}
    for vid in vehicle_ids:
        v_idx = vehicle_id2idx[vid]
        vid_path = root_path / vid
        for yaml_file in vid_path.glob("0*.yaml"):
            frame_prefix = yaml_file.name.split(".yaml")[0]
            frame_num_str = ''.join([c for c in frame_prefix if c.isdigit()])
            if not frame_num_str:
                print(f"error: {yaml_file}")
                continue
            f_idx = frame_num2idx[frame_num_str]  

            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    yaml_data = yaml.load(f, Loader=yaml.FullLoader)
                true_ego_pos = yaml_data.get("true_ego_pos", [])
                xyz = true_ego_pos[:3] if len(true_ego_pos) >= 3 else [np.nan, np.nan, np.nan]
                xyz = [float(val) if (val is not None and not np.isnan(val)) else np.nan for val in xyz]
                coord_dict[v_idx][f_idx] = xyz
            except Exception as e:
                print(f"error: {yaml_file}")
                coord_dict[v_idx][f_idx] = [np.nan, np.nan, np.nan]
    
    distance_tensor = tf.zeros([frame_num, vehicle_num, vehicle_num], dtype=tf.float32)
    distance_tensor = distance_tensor * tf.constant(np.nan, dtype=tf.float32)

    nlos_mask = tf.ones([frame_num, vehicle_num, vehicle_num], dtype=tf.float32)
    
    for f_idx in range(frame_num):
        for v1_idx in range(vehicle_num):
            for v2_idx in range(vehicle_num):
                if v1_idx == v2_idx:
                    distance_tensor = tf.tensor_scatter_nd_update(distance_tensor, [[f_idx, v1_idx, v2_idx]], [0.0])
                    nlos_mask = tf.tensor_scatter_nd_update(nlos_mask, [[f_idx, v1_idx, v2_idx]], [0.0])
                v1_xyz = coord_dict[v1_idx].get(f_idx, [np.nan, np.nan, np.nan])
                v2_xyz = coord_dict[v2_idx].get(f_idx, [np.nan, np.nan, np.nan])
                if any(np.isnan(v1_xyz)) or any(np.isnan(v2_xyz)):
                    continue 
                x1, y1, _ = v1_xyz
                x2, y2, _ = v2_xyz
                dx_abs = abs(x1 - x2) 
                dy_abs = abs(y1 - y2) 
                if dx_abs <= 20 or dy_abs <= 20:
                    current_nlos_flag = 0.0
                else:
                    current_nlos_flag = 1.0
                v1_xyz_tf = tf.convert_to_tensor(v1_xyz, dtype=tf.float32)
                v2_xyz_tf = tf.convert_to_tensor(v2_xyz, dtype=tf.float32)
                distance = tf.norm(v1_xyz_tf - v2_xyz_tf, ord=2)
                distance_tensor = tf.tensor_scatter_nd_update(distance_tensor, [[f_idx, v1_idx, v2_idx]], [distance])
                nlos_mask = tf.tensor_scatter_nd_update(nlos_mask, [[f_idx, v1_idx, v2_idx]], [current_nlos_flag])
    
    return distance_tensor, nlos_mask, frame_num2idx, vehicle_id2idx