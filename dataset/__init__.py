import torch
import numpy as np
from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from .dataset_nusc_surroundocc import NuScenes_Scene_SurroundOcc_Dataset
from .dataset_wrapper_nusc_occ import NuScenes_Scene_Occ_DatasetWrapper
from .dataset_nusc_surroundocc_stream import NuScenes_Scene_SurroundOcc_Dataset_Stream
from .dataset_wrapper_nusc_occ_stream import NuScenes_Scene_Occ_DatasetWrapper_Stream
from .dataset_nusc_surroundocc_streamtest import NuScenes_Scene_SurroundOcc_Dataset_StreamTest
from .dataset_nusc_surroundocc_mini import NuScenes_Scene_SurroundOcc_Dataset_Mini
from .dataset_wrapper_nusc_occ_mini import NuScenes_Scene_Occ_DatasetWrapper_Mini

def custom_collate_fn(data):
    """
    Custom collate function for data in the form of a list of triplets:
    [(imgs_1, [meta_dict_1], occ_labels_1), (imgs_2, [meta_dict_2], occ_labels_2), ...]
    
    Where each meta_dict contains scene information, projection matrices, etc.
    """
    if len(data) == 0:
        return [], [], []
    
    # Initialize lists to store batched data
    imgs_batch = []
    metas_batch = []
    occ_labels_batch = []
    
    # Iterate through each sample in the batch
    for imgs, meta_list, occ_labels in data:
        # Handle images (they're tensors)
        imgs_batch.append(imgs)
        
        # Handle metadata (list containing a single dictionary)
        # Keep the original structure but collect all meta dictionaries
        metas_batch.extend(meta_list)
        
        # Handle occupancy labels
        occ_labels_batch.append(occ_labels)
    
    # Stack images if they're tensors with the same shape
    if all(isinstance(img, torch.Tensor) for img in imgs_batch):
        try:
            imgs_batch = torch.stack(imgs_batch)
        except:
            # If images have different shapes, keep as list
            pass
    
    # Stack occupancy labels if they're tensors with the same shape
    if all(isinstance(label, torch.Tensor) for label in occ_labels_batch):
        try:
            occ_labels_batch = torch.stack(occ_labels_batch)
        except:
            # If labels have different shapes, keep as list
            pass
    
    return imgs_batch, metas_batch, occ_labels_batch


def build_dataloader(
            train_dataset_config,
            val_dataset_config,
            train_wrapper_config,
            val_wrapper_config,
            train_loader_config,
            val_loader_config,
            dist=False,
    ):
    train_dataset = OPENOCC_DATASET.build(train_dataset_config)
    val_dataset = OPENOCC_DATASET.build(val_dataset_config)

    train_wrapper = OPENOCC_DATAWRAPPER.build(train_wrapper_config, default_args={'in_dataset': train_dataset})
    val_wrapper = OPENOCC_DATAWRAPPER.build(val_wrapper_config, default_args={'in_dataset': val_dataset})

    train_sampler = val_sampler = None
    if dist:
        train_sampler = DistributedSampler(train_wrapper, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_wrapper, shuffle=False, drop_last=False)

    train_dataset_loader = DataLoader(dataset=train_wrapper,
                                    batch_size=train_loader_config["batch_size"],
                                    collate_fn=custom_collate_fn,
                                    shuffle=False if dist else train_loader_config["shuffle"],
                                    sampler=train_sampler,
                                    num_workers=train_loader_config["num_workers"],
                                    pin_memory=True)
    val_dataset_loader = DataLoader(dataset=val_wrapper,
                                    batch_size=val_loader_config["batch_size"],
                                    collate_fn=custom_collate_fn,
                                    shuffle=False if dist else val_loader_config["shuffle"],
                                    sampler=val_sampler,
                                    num_workers=val_loader_config["num_workers"],
                                    pin_memory=True)

    return train_dataset_loader, val_dataset_loader
