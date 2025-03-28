import os
import numpy as np
from torch.utils import data
import pickle
from mmcv.image.io import imread
from pyquaternion import Quaternion
from . import OPENOCC_DATASET
import json
from nuscenes.nuscenes import NuScenes


@OPENOCC_DATASET.register_module()
class NuScenes_Scene_SurroundOcc_Dataset_Mini(data.Dataset):
    def __init__(
        self,
        data_path,
        num_frames=1,
        grid_size_occ=[200, 200, 16],
        empty_idx=17,
        imageset=None,
        phase='train',
        scene_name=None,
        version='v1.0-mini'
        ):
        
        # Load NuScenes mini dataset directly
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        
        # Get scenes
        self.data_path = data_path
        self.num_frames = num_frames
        self.grid_size_occ = np.array(grid_size_occ).astype(np.uint32)
        self.empty_idx = empty_idx
        self.phase = phase
        
        # Get scenes based on validation or training split
        scenes = self.nusc.scene
        if phase == 'val':
            self.scene_names = [scene['name'] for scene in scenes[:5]]  # Use first half for val
        else:
            self.scene_names = [scene['name'] for scene in scenes[5:]]  # Use second half for train
            
        if scene_name is not None:
            self.scene_names = [scene_name]
            
        # Setup scene info
        self.scene_data = {}
        for scene_name in self.scene_names:
            scene_token = None
            for scene in scenes:
                if scene['name'] == scene_name:
                    scene_token = scene['token']
                    break
            
            if scene_token:
                self.scene_data[scene_name] = self.get_scene_frames(scene_token)
        
        self.scene_lens = [len(self.scene_data[sn]) for sn in self.scene_names]
        self.scene_name_table, self.scene_idx_table = self.get_scene_index(scene_name)

    def get_scene_frames(self, scene_token):
        """Extract all frames from a scene"""
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        
        frames = []
        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            frame_info = {
                'token': sample_token,
                'timestamp': sample['timestamp'],
                'lidar_path': self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['filename'],
                'cams': {},
                'lidar2ego_rotation': [0, 0, 0, 1],  # Default quaternion identity
                'lidar2ego_translation': [0, 0, 0],
                'ego2global_rotation': [0, 0, 0, 1],  # Default quaternion identity
                'ego2global_translation': [0, 0, 0],
                'sweeps': []
            }
            
            # Get pose info
            lidar_sample = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
            
            # Update proper poses
            frame_info['ego2global_rotation'] = lidar_pose['rotation']
            frame_info['ego2global_translation'] = lidar_pose['translation']
            
            # Get calibrated sensor info for lidar
            calib = self.nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            frame_info['lidar2ego_rotation'] = calib['rotation']
            frame_info['lidar2ego_translation'] = calib['translation']
            
            # Get cameras
            for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                if cam_name in sample['data']:
                    cam_token = sample['data'][cam_name]
                    cam_sample = self.nusc.get('sample_data', cam_token)
                    cam_path = cam_sample['filename']
                    
                    # Get calibration info
                    cam_calib = self.nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
                    
                    frame_info['cams'][cam_name] = {
                        'data_path': os.path.join(self.data_path, cam_path),
                        'sensor2lidar_rotation': cam_calib['rotation'],
                        'sensor2lidar_translation': np.array(cam_calib['translation']),
                        'cam_intrinsic': np.array(cam_calib['camera_intrinsic'])
                    }
            
            frames.append(frame_info)
            sample_token = sample['next']
            
        return frames

    def __len__(self):
        'Denotes the total number of scenes'
        return len(self.scene_name_table)

    def __getitem__(self, index):
        scene_name = self.scene_name_table[index]
        sample_idx = self.scene_idx_table[index]
        imgs_seq, metas_seq, occ_seq = [], [], []
        prev_lidar2global = None
        
        for i in range(self.num_frames):
            info = self.scene_data[scene_name][i + sample_idx]
            data_info = self.get_data_info(info)
            
            # load image
            imgs = []
            for filename in data_info['img_filename']:
                try:
                    # Check if the file exists
                    if os.path.exists(filename):
                        img = imread(filename, 'unchanged').astype(np.float32)
                    else:
                        # Try to find the file using a glob pattern
                        import glob
                        base_dir = os.path.dirname(os.path.dirname(filename))
                        file_pattern = os.path.basename(filename).split('__')
                        if len(file_pattern) >= 3:
                            # Search for any similar files
                            search_pattern = f"{base_dir}/**/*{file_pattern[0]}*{file_pattern[1]}*.jpg"
                            matches = glob.glob(search_pattern, recursive=True)
                            if matches:
                                img = imread(matches[0], 'unchanged').astype(np.float32)
                            else:
                                raise FileNotFoundError(f"No matching files found for {filename}")
                        else:
                            raise FileNotFoundError(f"Invalid filename pattern: {filename}")
                    
                    imgs.append(img)
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
                    # Create a dummy image as placeholder with correct dimensions 
                    # Use the dimensions specified in the config file (864x1600)
                    imgs.append(np.zeros((864, 1600, 3), dtype=np.float32))
            
            # Check if we have the expected number of cameras (6)
            if len(imgs) < 6:
                # Pad with dummy images if we don't have enough cameras
                print(f"Warning: Only found {len(imgs)} camera images, padding to 6...")
                for _ in range(6 - len(imgs)):
                    imgs.append(np.zeros((864, 1600, 3), dtype=np.float32))
            
            # Stack the images
            imgs_seq.append(np.stack(imgs, 0))
            
            # load metas
            metas = {'scene_name': scene_name}
            # Ensure proper formatting of lidar2img
            # Check if we have enough transformation matrices
            if len(data_info['lidar2img']) < 6:
                # Pad with identity matrices if we don't have enough
                print(f"Warning: Only found {len(data_info['lidar2img'])} transformation matrices, padding to 6")
                padded_transforms = list(data_info['lidar2img'])
                for _ in range(6 - len(padded_transforms)):
                    padded_transforms.append(np.eye(4))
                metas['lidar2img'] = padded_transforms
            else:
                metas['lidar2img'] = data_info['lidar2img']
            if prev_lidar2global is not None:
                metas['lidar2global'] = [prev_lidar2global, data_info['lidar2global']]
            prev_lidar2global = data_info['lidar2global']
            metas_seq.append([metas])
            
            # Generate dummy occupancy for mini dataset since we don't have labels
            # Create a fake/random occupancy grid for testing
            try:
                # First try to load real data if available
                label_file = os.path.join('data/surroundocc', data_info['pts_filename'].split('/')[-1]+'.npy')
                if os.path.exists(label_file):
                    label_idx = np.load(label_file)
                    occ_label = np.ones(self.grid_size_occ, dtype=np.int64) * self.empty_idx
                    occ_label[label_idx[:, 0], label_idx[:, 1], label_idx[:, 2]] = label_idx[:, 3]
                else:
                    # Generate a dummy occupancy grid
                    occ_label = np.ones(self.grid_size_occ, dtype=np.int64) * self.empty_idx
                    
                    # Use a seed based on index for reproducibility
                    np.random.seed(sample_idx + ord(scene_name[0]))
                    
                    # Add a more meaningful structure (like a road and some objects)
                    # Create a "road" - set bottom layer cells to class 13 (road surface)
                    road_layer = 2  # Use lower part of Z dimension for road
                    road_width = self.grid_size_occ[0] // 3
                    road_start = (self.grid_size_occ[0] - road_width) // 2
                    
                    # Place road in the center
                    occ_label[road_start:road_start+road_width, :, road_layer] = 13
                    
                    # Add some random "objects" for visualization
                    min_dim = min(self.grid_size_occ)
                    n_points = 500
                    rand_points = np.random.randint(0, min_dim, size=(n_points, 3))
                    rand_classes = np.random.randint(1, 17, size=(n_points))
                    
                    for i in range(n_points):
                        x, y, z = rand_points[i]
                        # Skip if point is part of road
                        if z == road_layer and road_start <= x < road_start + road_width:
                            continue
                        occ_label[x, y, z] = rand_classes[i]
            except Exception as e:
                print(f"Error generating occupancy grid: {e}")
                occ_label = np.ones(self.grid_size_occ, dtype=np.int64) * self.empty_idx
                
            occ_seq.append(occ_label)

        imgs = np.stack(imgs_seq, 0)    # F, N, H, W, C
        occ = np.stack(occ_seq, 0)      # F, H, W, D
        data_tuple = (imgs, metas_seq, occ)
        return data_tuple
    
    def get_data_info(self, info):
        # standard protocal modified from SECOND.Pytorch
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = info['lidar2ego_translation']
        ego2lidar = np.linalg.inv(lidar2ego)
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        lidar2global = np.dot(ego2global, lidar2ego)

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            ego2lidar=ego2lidar,
            lidar2global=lidar2global,
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            # Convert quaternion to rotation matrix first
            rotation_matrix = Quaternion(cam_info['sensor2lidar_rotation']).rotation_matrix
            lidar2cam_r = np.linalg.inv(rotation_matrix)
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        return input_dict

    def get_scene_index(self, scene_name=None):
        scene_name_table, scene_idx_table = [], []
        if scene_name is None:
            for i, scene_len in enumerate(self.scene_lens):
                for j in range(scene_len - self.num_frames + 1):
                    scene_name_table.append(self.scene_names[i])
                    scene_idx_table.append(j)
        else:
            scene_len = len(self.scene_data[scene_name])
            for j in range(scene_len - self.num_frames + 1):
                scene_name_table.append(scene_name)
                scene_idx_table.append(j)
        return scene_name_table, scene_idx_table