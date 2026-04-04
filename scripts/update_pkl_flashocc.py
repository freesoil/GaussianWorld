import pickle
import os

def update_pkl_for_flashocc():
    data_root = '/data'
    pkl_path = os.path.join(data_root, 'bevdetv2-nuscenes_infos_val_occ.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    for info in data['infos']:
        token = info['token']
        # Point to the directory containing labels.npz
        info['occ_path'] = os.path.join('data/nuscenes_occ_npz', token)
        
    out_path = os.path.join(data_root, 'bevdetv2-nuscenes_infos_val_flashocc.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Created {out_path}")

if __name__ == "__main__":
    update_pkl_for_flashocc()
