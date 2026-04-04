import numpy as np
import os
import glob
from tqdm import tqdm

def convert_surroundocc_to_occ3d():
    src_dir = '/data/data/surroundocc'
    dst_dir = '/data/data/nuscenes_occ_npz'
    os.makedirs(dst_dir, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(src_dir, "*.npy"))
    print(f"Converting {len(npy_files)} files...")
    
    for npy_file in tqdm(npy_files):
        data = np.load(npy_file)
        # Reconstruct 200x200x16 grid
        # SurroundOcc: (N, 4) -> (x, y, z, label)
        semantics = np.ones((200, 200, 16), dtype=np.uint8) * 17  # Default to empty
        semantics[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]
        
        # Occ3D expects 'semantics', 'mask_lidar', 'mask_camera'
        # We don't have real masks, so we use all ones or simple heuristic
        mask_lidar = np.ones((200, 200, 16), dtype=np.uint8)
        mask_camera = np.ones((200, 200, 16), dtype=np.uint8)
        
        # Save as npz
        out_filename = os.path.basename(npy_file).replace('.npy', '.npz')
        # We need to maintain the same filename (token)
        # Wait, FlashOCC looks for info['occ_path']/labels.npz
        # So we need one directory per sample?
        sample_dir = os.path.join(dst_dir, os.path.basename(npy_file).replace('.pcd.bin.npy', ''))
        os.makedirs(sample_dir, exist_ok=True)
        np.savez_compressed(os.path.join(sample_dir, 'labels.npz'), 
                            semantics=semantics,
                            mask_lidar=mask_lidar,
                            mask_camera=mask_camera)

if __name__ == "__main__":
    convert_surroundocc_to_occ3d()
