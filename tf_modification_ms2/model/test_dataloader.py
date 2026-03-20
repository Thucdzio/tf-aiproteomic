import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from model.dataloader import MS2Dataset
import os
import numpy as np  

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Sampler
import random

class LengthGroupedSampler(Sampler):
    def __init__(self, nAA_list, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Nhóm các index theo độ dài nAA
        # Ví dụ: {10: [1, 5, 20], 15: [2, 3, 4, 10...]}
        self.group_indices = {}
        for idx, nAA in enumerate(nAA_list):
            if nAA not in self.group_indices:
                self.group_indices[nAA] = []
            self.group_indices[nAA].append(idx)
        
        self.batches = []
        for nAA, indices in self.group_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Chia indices của cùng một độ dài thành các batch
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i + self.batch_size])
        
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def collate_fn_ms2_grouped(batch):
    # Vì tất cả b['aa_idx'] trong batch này đều cùng size, dùng torch.stack thay vì pad_sequence
    aa_idx = torch.stack([b['aa_idx'] for b in batch])
    mod_x = torch.stack([b['mod_x'] for b in batch])
    target = torch.stack([b['target'] for b in batch])
    
    return {
        "aa_idx": aa_idx,
        "mod_x": mod_x,
        "target": target,
        "charge": torch.stack([b['charge'] for b in batch]),
        "nce": torch.stack([b['nce'] for b in batch]),
        "instrument": torch.stack([b['instrument'] for b in batch]),
        "psm_id": [b['psm_id'] for b in batch]
    }

if __name__ == "__main__":
    base_dir = r"D:/DACN/tf-test/tf_modification_ms2/data/251128"
    index_df = pd.read_csv(r"D:/DACN/tf-test/tf_modification_ms2/data/_index.tsv", header=None)
    
    print(f"Đang nạp dữ liệu từ {len(index_df)} file...")
    dfs = []
    # Load 50 file để kiểm tra tính đa dạng của Batch
    for i, rel_path in enumerate(index_df[0][:50]): 
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            temp = pd.read_csv(full_path, sep="\t")
            temp["psm_id"] = f"psm_{i}"
            dfs.append(temp)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Khởi tạo Dataset (Đảm bảo MS2Dataset của mày đã được update __getitem__ để trả về nce, inst, charge)
    dataset = MS2Dataset(full_df)
    custom_sampler = LengthGroupedSampler(
        dataset.nAA_list, 
        batch_size=4, 
        shuffle=True
    )
    dataloader = DataLoader(
        dataset, 
        batch_sampler=custom_sampler, 
        collate_fn=collate_fn_ms2_grouped
    )

    # Lấy 1 batch ngẫu nhiên
    batch = next(iter(dataloader))
    
    print("\n" + "="*50)
    print("KIỂM TRA SHAPE CỦA BATCH")
    print("="*50)
    print(f"AA Indices shape:      {batch['aa_idx'].shape}   -> (Batch, Max_Len)")
    print(f"Mod Features shape:    {batch['mod_x'].shape}  -> (Batch, Max_Len, 109)")
    print(f"Target shape:          {batch['target'].shape}   -> (Batch, Max_Len, 8)")
    
    # KIỂM TRA CÁC THÔNG SỐ ĐIỀU KIỆN (MỚI)
    print(f"Charge shape:          {batch['charge'].shape}   -> (Batch, 1)")
    print(f"NCE shape:             {batch['nce'].shape}      -> (Batch, 1)")
    print(f"Instrument shape:      {batch['instrument'].shape} -> (Batch, 1)")

    # Kiểm tra chi tiết 1 mẫu trong batch
    idx_in_batch = 0
    psm_id = batch['psm_id'][idx_in_batch]
    
    print("\n" + "="*50)
    print(f"CHI TIẾT MẪU: {psm_id}")
    print("="*50)
    
    # 1. In thông số môi trường
    print(f"Precursor Charge: {batch['charge'][idx_in_batch].item()}")
    print(f"Normalized NCE:   {batch['nce'][idx_in_batch].item()} (Mặc định 0.3 cho NCE=30)")
    print(f"Instrument ID:    {batch['instrument'][idx_in_batch].item()} (0: QE)")

    # 2. In AA Indices
    aa_list = batch['aa_idx'][idx_in_batch].tolist()
    print(f"\nAA Indices (Full 40): {aa_list}")
    
    # 3. In Target (Chỉ in các dòng có dữ liệu ion)
    print("\nTarget (Fragment Ions):")
    target_np = batch['target'][idx_in_batch].numpy()
    # mask = np.any(target_np > 0, axis=1)
    # if np.any(mask):
    #     target_df = pd.DataFrame(target_np[mask], columns=[
    #         "b_z1", "b_z2", "y_z1", "y_z2", "b_loss_z1", "b_loss_z2", "y_loss_z1", "y_loss_z2"
    #     ])
    #     print(target_df)
    # else:
    #     print(">>> CẢNH BÁO: Target rỗng (toàn 0)!")
    target_df = pd.DataFrame(target_np, columns=[
        "b_z1", "b_z2", "y_z1", "y_z2", "b_loss_z1", "b_loss_z2", "y_loss_z1", "y_loss_z2"
    ])
    print(target_df)
    # 4. Kiểm tra Modification
    mod_intensity = torch.sum(batch['mod_x'][idx_in_batch], dim=-1)
    active_sites = torch.where(mod_intensity > 0)[0].tolist()
    print(f"\nVị trí có Modification trong Tensor: {active_sites}")

    print("\n" + "="*50)
    print("KIỂM TRA HOÀN TẤT - SẴN SÀNG TRAIN")
    print("="*50)
