#!/usr/bin/env python3
"""Load one mini split batch (dummy SurroundOcc) to verify PKL + paths + images.

Run from repo root:
  export NUSCENES_DATA_ROOT=/path/to/nuscenes
  export GAUSSIANWORLD_DUMMY_SURROUNDOCC=1
  python tools/gen_nuscenes_mini_temporal_pkl.py --out-dir data
  python tools/smoke_mini_dataset.py
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.chdir(REPO)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    os.environ.setdefault(
        'NUSCENES_DATA_ROOT',
        os.path.expanduser('~/reservoir/datasets/autonomy/nuscenes'),
    )
    os.environ['GAUSSIANWORLD_DUMMY_SURROUNDOCC'] = '1'

    train_pkl = os.path.join(REPO, 'data', 'nuscenes_mini_temporal_infos_train.pkl')
    if not os.path.isfile(train_pkl):
        print('Missing', train_pkl, '- running gen_nuscenes_mini_temporal_pkl.py ...')
        subprocess.check_call(
            [sys.executable, os.path.join(REPO, 'tools', 'gen_nuscenes_mini_temporal_pkl.py'),
             '--dataroot', os.environ['NUSCENES_DATA_ROOT'],
             '--out-dir', os.path.join(REPO, 'data')],
        )

    from dataset import OPENOCC_DATASET, OPENOCC_DATAWRAPPER

    ds = OPENOCC_DATASET.build(dict(
        type='NuScenes_Scene_SurroundOcc_Dataset',
        data_path='data/nuscenes/',
        num_frames=1,
        offset=0,
        imageset='data/nuscenes_mini_temporal_infos_train.pkl',
        phase='train',
    ))
    wrap = OPENOCC_DATAWRAPPER.build(
        dict(
            type='NuScenes_Scene_Occ_DatasetWrapper',
            final_dim=[864, 1600],
            resize_lim=[1.0, 1.0],
            flip=False,
            phase='val',
        ),
        default_args={'in_dataset': ds},
    )
    print('dataset len:', len(wrap))
    imgs, metas, occ = wrap[0]
    print('imgs', imgs.shape, 'occ', occ.shape, 'ok')


if __name__ == '__main__':
    main()
