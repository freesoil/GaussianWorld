import pickle
import os
import sys
import numpy as np
# Hack for numpy 2.0 compatibility
try:
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
except ImportError:
    pass

from collections import defaultdict

def prepare_mini():
    data_root = '/data'
    gw_data_root = '/gw_data'
    
    # Load mini tokens for FlashOCC/MambaOCC
    with open(os.path.join(data_root, 'bevdetv2-nuscenes_infos_val.pkl'), 'rb') as f:
        mini_data_val = pickle.load(f)
    
    with open(os.path.join(data_root, 'bevdetv2-nuscenes_infos_train.pkl'), 'rb') as f:
        mini_data_train = pickle.load(f)
    
    # Load SurroundOcc full infos
    with open(os.path.join(data_root, 'nuscenes_infos_val.pkl'), 'rb') as f:
        full_val = pickle.load(f)
    
    with open(os.path.join(data_root, 'nuscenes_infos_train.pkl'), 'rb') as f:
        full_train = pickle.load(f)
        
    # Map token to occ_path
    token_to_occ = {}
    for info in full_val['infos']:
        token_to_occ[info['token']] = info['occ_path']
    for info in full_train['infos']:
        token_to_occ[info['token']] = info['occ_path']
        
    # 1. Update FlashOCC/MambaOCC mini pkls with occ_path
    print("Updating FlashOCC/MambaOCC mini pkls...")
    for info in mini_data_val['infos']:
        info['occ_path'] = token_to_occ.get(info['token'], "")
    for info in mini_data_train['infos']:
        info['occ_path'] = token_to_occ.get(info['token'], "")
        
    # Save with protocol 4 for compatibility
    with open(os.path.join(data_root, 'bevdetv2-nuscenes_infos_val_occ.pkl'), 'wb') as f:
        pickle.dump(mini_data_val, f, protocol=4)
    with open(os.path.join(data_root, 'bevdetv2-nuscenes_infos_train_occ.pkl'), 'wb') as f:
        pickle.dump(mini_data_train, f, protocol=4)
        
    # 2. Update GaussianWorld mini pkls with occ_path
    print("Updating GaussianWorld mini pkls...")
    
    for split in ['train', 'val']:
        filename = f'nuscenes_mini_temporal_infos_{split}.pkl'
        filepath = os.path.join(gw_data_root, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filepath} (not found)")
            continue
            
        with open(filepath, 'rb') as f:
            gw_data = pickle.load(f)
            
        # gw_data['infos'] is a dict grouped by scene
        for scene_name, scene_infos in gw_data['infos'].items():
            for info in scene_infos:
                info['occ_path'] = token_to_occ.get(info['token'], "")
                
        out_filename = f'nuscenes_temporal_infos_{split}_occ.pkl'
        out_filepath = os.path.join(gw_data_root, out_filename)
        with open(out_filepath, 'wb') as f:
            pickle.dump(gw_data, f, protocol=4)
        print(f"Created {out_filepath}")

if __name__ == "__main__":
    prepare_mini()
