import numpy as np
import torch
from torch.utils import data
from . import OPENOCC_DATAWRAPPER
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, ImageAug3D


# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

@OPENOCC_DATAWRAPPER.register_module()
class NuScenes_Scene_Occ_DatasetWrapper_Stream(data.Dataset):
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], flip=False, phase='train'):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, flip=flip, is_train=True),
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, flip=False, is_train=False),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            data = self.dataset[index]
            imgs, metas, occ = data

            # deal with img augmentation
            F, N, H, W, C = imgs.shape
            imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}
            
            try:
                for t in self.transforms:
                    imgs_dict = t(imgs_dict)
                imgs = imgs_dict['img']
                imgs = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)
                FN, C, H, W = imgs.shape
                imgs = imgs.reshape(F, N, C, H, W)
            except Exception as e:
                print(f"Error during image transformation: {e}, creating default images")
                # Create default normalized images
                imgs = np.zeros((F, N, 3, 864, 1600), dtype=np.float32)
                
                # Apply normalization to match what NormalizeMultiviewImage would do
                for f in range(F):
                    for n in range(N):
                        for c in range(3):
                            imgs[f, n, c] = (0 - img_norm_cfg['mean'][c]) / img_norm_cfg['std'][c]
                
                # Create default img_shape
                if 'img_shape' not in imgs_dict:
                    imgs_dict['img_shape'] = [(864, 1600, 3) for _ in range(F*N)]

            # Handle img_aug_matrix
            img_aug_matrix = None
            if imgs_dict.get('img_aug_matrix'):
                try:
                    img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0).reshape(F, N, 4, 4)
                except Exception as e:
                    print(f"Error processing img_aug_matrix: {e}, using identity matrices")
                    img_aug_matrix = np.tile(np.eye(4)[None, None, :, :], (F, N, 1, 1))
                    
            # Ensure metas has the right structure
            for i in range(F):
                if i >= len(metas) or not isinstance(metas[i], list) or len(metas[i]) == 0:
                    print(f"Missing metadata for frame {i}, creating default")
                    if i >= len(metas):
                        metas.append([{}])
                    elif not isinstance(metas[i], list):
                        metas[i] = [metas[i] if isinstance(metas[i], dict) else {}]
                    elif len(metas[i]) == 0:
                        metas[i].append({})
                
                # Add img_shape
                try:
                    metas[i][0]['img_shape'] = imgs_dict['img_shape'][6*i:6*(i+1)]
                except (KeyError, IndexError) as e:
                    print(f"Error setting img_shape: {e}, using default")
                    metas[i][0]['img_shape'] = [(864, 1600, 3) for _ in range(6)]
                    
                # Add img_aug_matrix if available
                if img_aug_matrix is not None:
                    metas[i][0]['img_aug_matrix'] = img_aug_matrix[i:i+1]
        except Exception as e:
            print(f"Critical error in dataset_wrapper: {e}, creating default data")
            # Create completely default data
            F, N = 1, 6  # Assume 1 frame, 6 cameras
            imgs = np.zeros((F, N, 3, 864, 1600), dtype=np.float32)
            metas = [[{'scene_name': 'error', 'img_shape': [(864, 1600, 3) for _ in range(N)]}]]
            occ = np.ones((F, 200, 200, 16), dtype=np.int64) * 17  # empty_idx=17
        
        data_tuple = (imgs, metas, occ)
        return data_tuple