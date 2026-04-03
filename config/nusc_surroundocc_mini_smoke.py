# nuScenes mini smoke train: temporal PKLs from tools/gen_nuscenes_mini_temporal_pkl.py
# and GAUSSIANWORLD_DUMMY_SURROUNDOCC=1 (no SurroundOcc .npy labels).
_base_ = './nusc_surroundocc_base.py'

max_epochs = 1
print_freq = 5

train_dataset_config = dict(
    type='NuScenes_Scene_SurroundOcc_Dataset',
    data_path='data/nuscenes/',
    num_frames=1,
    offset=0,
    imageset='data/nuscenes_mini_temporal_infos_train.pkl',
    phase='train',
)

val_dataset_config = dict(
    type='NuScenes_Scene_SurroundOcc_Dataset',
    data_path='data/nuscenes/',
    num_frames=1,
    offset=0,
    imageset='data/nuscenes_mini_temporal_infos_val.pkl',
    phase='val',
)

train_loader_config = dict(batch_size=1, shuffle=True, num_workers=0)
val_loader_config = dict(batch_size=1, shuffle=False, num_workers=0)
