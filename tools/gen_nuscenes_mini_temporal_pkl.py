#!/usr/bin/env python3
"""Build GaussianWorld temporal PKLs from nuScenes v1.0-mini (no mmcv/mmdet required).

Output format matches dataset expectation: {'infos': {scene_name: [info, ...], ...}}

Usage:
  export NUSCENES_DATA_ROOT=/path/to/nuscenes   # contains v1.0-mini/
  python tools/gen_nuscenes_mini_temporal_pkl.py --out-dir data
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
from collections import defaultdict

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion


CAMERA_TYPES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def _rel(root: str, path: str) -> str:
    path = str(path)
    root = osp.abspath(root)
    if osp.isabs(path):
        return osp.relpath(path, root)
    return path


def obtain_sensor2top(
    nusc,
    sensor_token,
    l2e_t,
    l2e_r_mat,
    e2g_t,
    e2g_r_mat,
    root: str,
    sensor_type: str = 'lidar',
):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = _rel(root, nusc.get_sample_data_path(sd_rec['token']))
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = np.array(sweep['sensor2ego_translation'], dtype=np.float64)
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = np.array(sweep['ego2global_translation'], dtype=np.float64)

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep


def sample_to_info(nusc, sample, root: str, max_sweeps: int = 10) -> dict:
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    lidar_path, _, _ = nusc.get_sample_data(lidar_token)

    l2e_t = np.array(cs_record['translation'], dtype=np.float64)
    l2e_r = cs_record['rotation']
    e2g_t = np.array(pose_record['translation'], dtype=np.float64)
    e2g_r = pose_record['rotation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    info = {
        'lidar_path': _rel(root, lidar_path),
        'token': sample['token'],
        'sweeps': [],
        'cams': {},
        'lidar2ego_translation': cs_record['translation'],
        'lidar2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sample['timestamp'],
    }

    for cam in CAMERA_TYPES:
        cam_token = sample['data'][cam]
        _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(
            nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, root, sensor_type=cam)
        cam_info['cam_intrinsic'] = cam_intrinsic
        info['cams'][cam] = cam_info

    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sweeps = []
    while len(sweeps) < max_sweeps:
        if sd_rec['prev'] == '':
            break
        sweep = obtain_sensor2top(
            nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, root, 'lidar')
        sweeps.append(sweep)
        sd_rec = nusc.get('sample_data', sd_rec['prev'])
    info['sweeps'] = sweeps
    return info


def ordered_samples_in_scene(nusc, scene_token: str):
    scene = nusc.get('scene', scene_token)
    out = []
    tok = scene['first_sample_token']
    while tok:
        out.append(nusc.get('sample', tok))
        tok = out[-1]['next']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--dataroot',
        default=os.environ.get('NUSCENES_DATA_ROOT', ''),
        help='nuScenes root (directory containing v1.0-mini/)',
    )
    ap.add_argument('--out-dir', default='data', help='GaussianWorld data/ output directory')
    ap.add_argument('--version', default='v1.0-mini')
    args = ap.parse_args()
    if not args.dataroot:
        ap.error('Pass --dataroot or set NUSCENES_DATA_ROOT')
    root = osp.abspath(args.dataroot)
    os.makedirs(args.out_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=root, verbose=False)

    train_names = set(splits.mini_train)
    val_names = set(splits.mini_val)

    train_infos = defaultdict(list)
    val_infos = defaultdict(list)

    for scene in nusc.scene:
        name = scene['name']
        if name not in train_names and name not in val_names:
            continue
        target = train_infos if name in train_names else val_infos
        for sample in ordered_samples_in_scene(nusc, scene['token']):
            target[name].append(sample_to_info(nusc, sample, root))

    def dump(path, infos_map):
        payload = {'infos': dict(infos_map)}
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        n_scenes = len(infos_map)
        n_frames = sum(len(v) for v in infos_map.values())
        print(f'Wrote {path} ({n_scenes} scenes, {n_frames} samples)')

    dump(osp.join(args.out_dir, 'nuscenes_mini_temporal_infos_train.pkl'), train_infos)
    dump(osp.join(args.out_dir, 'nuscenes_mini_temporal_infos_val.pkl'), val_infos)


if __name__ == '__main__':
    main()
