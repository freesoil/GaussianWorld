import numpy as np
import os
import glob
import pickle
from tqdm import tqdm

def convert_surroundocc_to_occ3d_mini():
    data_root = '/data'
    src_dir = os.path.join(data_root, 'data/surroundocc')
    dst_dir = os.path.join(data_root, 'data/nuscenes_occ_npz')
    os.makedirs(dst_dir, exist_ok=True)
    
    # Load mini tokens
    with open(os.path.join(data_root, 'bevdetv2-nuscenes_infos_val.pkl'), 'rb') as f:
        mini_data_val = pickle.load(f)
    # Token to lidar_filename mapping
    token_to_filename = {info['token']: os.path.basename(info['lidar_path']) for info in mini_data_val['infos']}
    
    print(f"Converting {len(token_to_filename)} files for mini validation...")
    
    for token, lidar_filename in tqdm(token_to_filename.items()):
        npy_filename = lidar_filename + '.npy'
        npy_file = os.path.join(src_dir, npy_filename)
        if not os.path.exists(npy_file):
            print(f"Warning: {npy_file} not found")
            continue
            
        data = np.load(npy_file)
        # Reconstruct 200x200x16 grid
        semantics = np.ones((200, 200, 16), dtype=np.uint8) * 17
        semantics[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]
        
        mask_lidar = np.ones((200, 200, 16), dtype=np.uint8)
        mask_camera = np.ones((200, 200, 16), dtype=np.uint8)
        
        # FlashOCC expects info['occ_path']/labels.npz
        # We'll set occ_path to 'data/nuscenes_occ_npz/<token>'
        sample_dir = os.path.join(dst_dir, token)
        os.makedirs(sample_dir, exist_ok=True)
        np.savez_compressed(os.path.join(sample_dir, 'labels.npz'), 
                            semantics=semantics,
                            mask_lidar=mask_lidar,
                            mask_camera=mask_camera)

if __name__ == "__main__":
    convert_surroundocc_to_occ3d_mini()
