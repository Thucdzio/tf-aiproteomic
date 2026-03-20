import numpy as np
import pandas as pd
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader

from model.featurize import get_batch_aa_indices, get_batch_mod_feature
from model.settings import MOD_DF
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
    
    charge = torch.stack([b['charge'] for b in batch]).view(-1, 1)
    nce = torch.stack([b['nce'] for b in batch]).view(-1, 1)
    inst = torch.stack([b['instrument'] for b in batch]).view(-1).long()

    return {
        "aa_idx": aa_idx,
        "mod_x": mod_x,
        "target": target,
        "charge": charge,
        "nce": nce,
        "instrument": inst,
        "psm_id": [b['psm_id'] for b in batch]
    }

class MS2Dataset(Dataset):
    def __init__(self, df):
        # Không cần max_len nữa vì ta sẽ dùng dynamic batching
        self.unimod_to_name = MOD_DF[MOD_DF['unimod_id'] != 0].set_index('unimod_id')['mod_name'].to_dict()
        
        # 1. Tiền xử lý Metadata
        self.df_meta = df.drop_duplicates('psm_id').copy()
        parsed_data = self.df_meta['ModifiedPeptide'].apply(self._parse_peptdeep_format)
        self.df_meta[['sequence', 'mods', 'mod_sites', 'nAA']] = pd.DataFrame(
            parsed_data.tolist(), index=self.df_meta.index
        )
        self.df_meta.set_index('psm_id', inplace=True)
        self.default_nce = 30.0 / 100.0
        self.default_inst = 0

        max_nAA_dataset = self.df_meta['nAA'].max()

        # 2. Xử lý Input (Giữ nguyên thô, không pad)
        # Lưu ý: get_batch_aa_indices trả về L+2
        self.aa_indices_dict = get_batch_aa_indices(self.df_meta['sequence'].values.astype('U'))
        self.aa_indices_dict = np.where((self.aa_indices_dict >= 1) & (self.aa_indices_dict <= 26), self.aa_indices_dict, 0)
        
        # Xử lý Mod Features thô
        feat_df = self.df_meta.reset_index().copy()
        feat_df['nAA'] = max_nAA_dataset
        self.mod_x_data = get_batch_mod_feature(feat_df) 
        
        # 3. Xử lý Target (Chỉ lấy đúng nAA - 1 dòng cho mỗi peptide)
        self.targets = self._process_targets_dynamic(df)
        self.psm_ids = self.df_meta.index.values 
        self.charges = self.df_meta['PrecursorCharge'].values
        self.nAA_list = self.df_meta['nAA'].values

    def _process_targets_dynamic(self, df):
        frag_types = ["b_z1", "b_z2", "y_z1", "y_z2", "b_modloss_z1", "b_modloss_z2", "y_modloss_z1", "y_modloss_z2"]
        df = df.copy()
        df['charge_num'] = df['FragmentCharge'].astype(str).str.extract('(\d+)').astype(int)
        df['loss_clean'] = df['FragmentLossType'].apply(lambda x: '' if x in ['noloss', ''] else '_modloss')
        df['target_col'] = df['FragmentType'] + df['loss_clean'] + '_z' + df['charge_num'].astype(str)
        
        pivot = df.pivot_table(index=['psm_id', 'FragmentNumber'], columns='target_col', values='RelativeIntensity', aggfunc='max')
        pivot = pivot.reindex(columns=frag_types, fill_value=0.0).fillna(0)
        
        target_dict = {}
        for psm_id, group in pivot.groupby(level=0):

            nAA = self.df_meta.loc[psm_id, 'nAA'] 
            full_index = pd.MultiIndex.from_product([[psm_id], range(1, nAA)], names=['psm_id', 'FragmentNumber'])
            actual_matrix = group.reindex(full_index, fill_value=0.0)
            target_dict[psm_id] = actual_matrix.values.astype(np.float32)

        return target_dict

    def _parse_peptdeep_format(self, mod_peptide):
        peptide = mod_peptide.strip('_')
        clean_seq = re.sub(r'\(UniMod:\d+\)', '', peptide)
        nAA = len(clean_seq)
        mods, mod_sites = [], []

        n_term_match = re.match(r'^\((UniMod:(\d+))\)', peptide)
        if n_term_match:
            uid = int(n_term_match.group(2))
            mods.append("Acetyl@Any_N-term" if uid == 1 else MOD_DF[(MOD_DF.unimod_id == uid) & (MOD_DF.mod_name.str.contains('N-term'))].mod_name.iloc[0])
            mod_sites.append(0)
            peptide = peptide[len(n_term_match.group(0)):]

        residue_matches = re.finditer(r'([A-Z])\((UniMod:(\d+))\)', peptide)
        for m in residue_matches:
            aa, uid = m.group(1), int(m.group(3))
            possible_names = MOD_DF[(MOD_DF.unimod_id == uid) & (MOD_DF.mod_name.str.endswith(f"@{aa}"))].mod_name.values
            mods.append(possible_names[0] if len(possible_names) > 0 else MOD_DF[MOD_DF.unimod_id == uid].mod_name.iloc[0])
            pre_str = peptide[:m.start(1)+1]
            mod_sites.append(len(re.sub(r'\(UniMod:\d+\)', '', pre_str)))

        return clean_seq, ";".join(mods), ";".join(map(str, mod_sites)), nAA

    def __len__(self): return len(self.psm_ids)

    def __getitem__(self, idx):
        psm_id = self.psm_ids[idx]
        nAA = self.nAA_list[idx]
        
       
        aa_idx = torch.tensor(self.aa_indices_dict[idx, :nAA+2], dtype=torch.long)
        mod_x = torch.tensor(self.mod_x_data[idx, :nAA+2, :], dtype=torch.float32)
        target = torch.tensor(self.targets[psm_id], dtype=torch.float32)
        
        return {
            "aa_idx": aa_idx,
            "mod_x": mod_x,
            "charge": torch.tensor([self.charges[idx]], dtype=torch.float32),
            "nce": torch.tensor([self.default_nce], dtype=torch.float32),
            "instrument": torch.tensor([self.default_inst], dtype=torch.long),
            "target": target,
            "psm_id": psm_id
        }
    


if __name__ == "__main__":
    base_dir = r"D:/DACN/tf-test/tf_modification_ms2/data/251128"
    index_path = r"D:/DACN/tf-test/tf_modification_ms2/data/_index.tsv"
    
    # 1. Load toàn bộ index
    index_df = pd.read_csv(index_path, header=None)
    
    print(f"Đang nạp {len(index_df)} file dữ liệu...")
    dfs = []
    for i, rel_path in enumerate(index_df[0][:]):
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            temp = pd.read_csv(full_path, sep="\t")
            temp["psm_id"] = f"psm_{i}"
            dfs.append(temp)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 2. Khởi tạo Dataset (max_len=40 để khớp Transformer)
    dataset = MS2Dataset(full_df)
    
    # 3. Ghi log kiểm tra ra file txt
    log_file = "debug_modification_check.txt"
    print(f"Bắt đầu kiểm tra và ghi log vào: {log_file}...")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== KẾT QUẢ KIỂM TRA MODIFICATION (Tổng số: {len(dataset)} mẫu) ===\n")
        f.write("-" * 80 + "\n")
        
        # Duyệt qua toàn bộ dataset
        for i in range(len(dataset)):
            sample = dataset[i]
            psm_id = sample['psm_id']
            meta = dataset.df_meta.loc[psm_id]
            
            # Kiểm tra modification trong Tensor
            mod_intensity = torch.sum(sample['mod_x'], dim=-1)
            active_sites = torch.where(mod_intensity > 0)[0].tolist()
            
            # Ghi thông tin cơ bản
            f.write(f"[{i}] PSM ID: {psm_id}\n")
            f.write(f"Peptide Gốc: {full_df[full_df['psm_id']==psm_id]['ModifiedPeptide'].iloc[0]}\n")
            f.write(f"Chuỗi AA: {meta['sequence']}\n")
            f.write(f"Mod AlphaBase: {meta['mods']}\n")
            f.write(f"Vị trí Site: {meta['mod_sites']}\n")
            f.write(f"Indices: {sample['aa_idx'][:meta['nAA']+2].tolist()}\n")
            f.write(f"Vị trí Mod trong Tensor: {active_sites}\n")
            
            # Cảnh báo nếu có mod nhưng Tensor rỗng
            if len(str(meta['mods'])) > 0 and len(active_sites) == 0:
                f.write(">>> CẢNH BÁO: PHÁT HIỆN MOD NHƯNG TENSOR MOD_X RỖNG! <<<\n")
            
            f.write("-" * 40 + "\n")
            
            # In tiến độ ra console sau mỗi 100 mẫu để mày biết nó đang chạy
        

    print(f"\n[Xong!]  Mở file '{log_file}' lên để xem kết quả chi tiết.")