import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from model.dataloader import MS2Dataset
import os
import numpy as np  

if __name__ == "__main__":
    base_dir = r"D:/DACN/tf-test/tf_modification_ms2/data/251128"
    index_df = pd.read_csv(r"D:/DACN/tf-test/tf_modification_ms2/data/_index.tsv", header=None)
    
    print(f"Đang nạp dữ liệu từ {len(index_df)} file...")
    dfs = []
    for i, rel_path in enumerate(index_df[0][:50]): # Load 50 file để test batch
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            temp = pd.read_csv(full_path, sep="\t")
            temp["psm_id"] = f"psm_{i}"
            dfs.append(temp)
    
    full_df = pd.concat(dfs, ignore_index=True)
    dataset = MS2Dataset(full_df, max_len=40)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Lấy 1 batch ngẫu nhiên
    batch = next(iter(dataloader))
    
    print("\n" + "="*50)
    print("KIỂM TRA SHAPE CỦA BATCH")
    print("="*50)
    print(f"AA Indices shape:  {batch['aa_idx'].shape}  -> (Batch, Max_Len)")
    print(f"Mod Features shape: {batch['mod_x'].shape} -> (Batch, Max_Len, 109)")
    print(f"Target shape:       {batch['target'].shape}  -> (Batch, Max_Len, 8)")

    # Kiểm tra chi tiết 1 mẫu trong batch
    idx_in_batch = 0
    psm_id = batch['psm_id'][idx_in_batch]
    
    print("\n" + "="*50)
    print(f"CHI TIẾT MẪU: {psm_id}")
    print("="*50)
    
    # In AA Indices (xem có bị toàn số 0 không)
    print(f"AA Indices (đầu chuỗi): {batch['aa_idx'][idx_in_batch][:].tolist()}")
    
    # In Target (Ma trận MS2 8 cột)
    print("\n Target :")
    target_np = batch['target'][idx_in_batch].numpy()
    # Lọc những dòng có giá trị > 0 để xem cho rõ
    mask = np.any(target_np > 0, axis=1)
    target_df = pd.DataFrame(target_np[mask][:], columns=[
        "b_z1", "b_z2", "y_z1", "y_z2", "b_loss_z1", "b_loss_z2", "y_loss_z1", "y_loss_z2"
    ])
    print(target_df)

    # Kiểm tra Mod
    mod_intensity = torch.sum(batch['mod_x'][idx_in_batch], dim=-1)
    active_sites = torch.where(mod_intensity > 0)[0].tolist()
    print(f"\nVị trí có Modification trong Tensor: {active_sites}")