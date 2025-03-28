import torch
import numpy as np
from . import OPENOCC_DATAWRAPPER
import cv2
import random


@OPENOCC_DATAWRAPPER.register_module()
class NuScenes_Scene_Occ_DatasetWrapper_Mini(torch.utils.data.Dataset):
    def __init__(
        self,
        in_dataset,  # Changed parameter name to match original wrapper
        final_dim,
        resize_lim,
        flip,
        image_normalizer=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        phase='train',
    ):
        self.dataset = in_dataset  # Store it as self.dataset internally
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.flip = flip
        self.phase = phase
        self.image_normalizer = image_normalizer
        self.image_std = image_std
        print(f"Initialized dataset wrapper with final_dim={final_dim}")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        print(f"Loading dataset item at index {index}")
        print(f"Dataset type: {type(self.dataset)}")
        # Check dataset length
        print(f"Dataset length: {len(self.dataset)}")
        # Get data from dataset
        data = self.dataset[index]
        imgs, metas, occ_label = data
        
        print(f'imgs: {imgs.shape}')
        print(f'metas: {metas}')
        print(f"Type of metas: {type(metas)}")
        print(f'occ_label: {occ_label.shape}')
        
        if isinstance(metas, list):
            print(f"Length of metas list: {len(metas)}")
            if len(metas) > 0:
                print(f"Type of first element: {type(metas[0])}")
        
        # Create a new array with the correct dimensions instead of modifying in place
        F, N = imgs.shape[0], imgs.shape[1]  # Frames, Cameras
        target_h, target_w = self.final_dim
        
        print(f"Creating transformed imgs array with shape: ({F}, {N}, {target_h}, {target_w}, 3)")
        transformed_imgs = np.zeros((F, N, target_h, target_w, 3), dtype=np.float32)
        
        # Resize and normalize images
        for i in range(F):  # For each frame
            for j in range(N):  # For each camera
                transformed_imgs[i, j] = self.img_transform(imgs[i, j])
        
        # Create frames of images in the right format C,H,W
        print(f"Reshaping to CHW format")
        imgs_reshaped = []
        for i in range(F):
            frame_imgs = []
            for j in range(N):
                # Transpose from H,W,C to C,H,W
                img_chw = transformed_imgs[i, j].transpose(2, 0, 1)
                frame_imgs.append(img_chw)
            imgs_reshaped.append(np.stack(frame_imgs, axis=0))
                
        # Stack frames
        imgs_reshaped = np.stack(imgs_reshaped, axis=0)  # Shape: (F, N, C, H, W)
        print(f"Reshaped imgs shape: {imgs_reshaped.shape}")
        
        # Convert to PyTorch tensors
        imgs = torch.from_numpy(imgs_reshaped).float()
        print(f"Final tensor shape: {imgs.shape}")
        
        # Convert occupancy label to tensor
        occ_label = torch.from_numpy(occ_label).long()
        
        # Setup metadata
        print(f"Setting up metadata")
        H, W = target_h, target_w
        C = 3
        
        # Create img_shape array for each camera
        img_shapes = [(H, W, C) for _ in range(N)]
        
        # Create transformation matrices
        img_aug_matrices = []
        for j in range(N):
            img_aug_matrices.append(np.eye(4))  # Identity matrix per camera
            
        # Stack matrices in the right shape: (1, N, 4, 4)
        img_aug_matrix = np.stack(img_aug_matrices).reshape(1, N, 4, 4)
        
        # Update metas dictionary directly, following the original format
        # Handle metas based on the format from the dataset
        if isinstance(metas, list) and len(metas) > 0 and isinstance(metas[0], list) and len(metas[0]) > 0:
            # This is likely in the format from the dataset: list of list of dicts
            # Extract the first frame's metadata
            frame_metas = metas[0]
            
            # Assume the first element in frame_metas is a dict
            if isinstance(frame_metas[0], dict):
                meta_dict = frame_metas[0]
                meta_dict['img_shape'] = img_shapes
                meta_dict['img_aug_matrix'] = img_aug_matrix
                
                # Keep metas in the same format
                metas = metas
            else:
                print(f"Unexpected format for frame_metas[0]: {type(frame_metas[0])}")
                # Create a fallback dictionary
                meta_dict = {
                    'scene_name': 'unknown',
                    'img_shape': img_shapes,
                    'img_aug_matrix': img_aug_matrix
                }
                metas = [[meta_dict]]
        else:
            print(f"Metas is not in expected format: {type(metas)}")
            # Create a new metas in the expected format
            meta_dict = {
                'scene_name': 'unknown',
                'img_shape': img_shapes,
                'img_aug_matrix': img_aug_matrix
            }
            if isinstance(metas, dict):
                # Add required fields
                metas['img_shape'] = img_shapes
                metas['img_aug_matrix'] = img_aug_matrix
            else:
                # Create a new metas list
                metas = [[meta_dict]]
        
        print(f"Final metas type: {type(metas)}")
        if isinstance(metas, list):
            print(f"Metas list length: {len(metas)}")
            
        # Return data in the same format as original dataset
        print(f"Returning data tuple")
        print(f"imgs: {imgs.shape}")
        print(f"metas: {metas}")
        print(f"occ_label: {occ_label.shape}")
        return (imgs, metas, occ_label)
        
    def img_transform(self, img):
        # Get target dimensions
        target_h, target_w = self.final_dim
        
        # Skip resizing and directly resize to target dimensions
        # This avoids the shape mismatch issues
        h, w, c = img.shape
        # Direct resize to target dimensions
        if h != target_h or w != target_w:
            img = cv2.resize(img, (target_w, target_h))
        
        # Normalize
        img = img / 255.0
        img = (img - np.array(self.image_normalizer)) / np.array(self.image_std)
        
        return img