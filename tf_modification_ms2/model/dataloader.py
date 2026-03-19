import numpy as np
import pandas as pd
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader

from model.featurize import get_batch_aa_indices, get_batch_mod_feature
from model.settings import MOD_DF

class MS2Dataset(Dataset):
    def __init__(self, df, max_len=30):
        self.max_len = max_len
        # Load từ điển UniMod ID -> Tên chuẩn AlphaBase (vd: 21 -> Phospho@S)
        self.unimod_to_name = MOD_DF[MOD_DF['unimod_id'] != 0].set_index('unimod_id')['mod_name'].to_dict()
        
        # 1. Tiền xử lý Metadata
        self.df_meta = df.drop_duplicates('psm_id').copy()
        parsed_data = self.df_meta['ModifiedPeptide'].apply(self._parse_peptdeep_format)
        self.df_meta[['sequence', 'mods', 'mod_sites', 'nAA']] = pd.DataFrame(
            parsed_data.tolist(), index=self.df_meta.index
        )
        
        # QUAN TRỌNG: Đặt psm_id làm index để tránh lỗi KeyError khi dùng .loc
        self.df_meta.set_index('psm_id', inplace=True)

        # 2. Xử lý Input
        raw_aa = get_batch_aa_indices(self.df_meta['sequence'].values.astype('U'))
        raw_aa = np.where((raw_aa >= 1) & (raw_aa <= 26), raw_aa, 0)
        
        feat_df = self.df_meta.reset_index().copy()
        actual_max_nAA = feat_df['nAA'].max()
        feat_df['nAA'] = actual_max_nAA

        raw_mod = get_batch_mod_feature(feat_df) # Reset để hàm feature đọc được psm_id nếu cần
        
        self.aa_indices = self._pad_or_clip_2d(raw_aa, max_len)
        self.mod_x = self._pad_or_clip_3d(raw_mod, max_len)
        
        # 3. Xử lý Target
        self.targets = self._process_targets(df)
        self.psm_ids = self.df_meta.index.values # Lấy từ index đã set
        self.charges = self.df_meta['PrecursorCharge'].values

    def _parse_peptdeep_format(self, mod_peptide):
        peptide = mod_peptide.strip('_')
        
        # Tách chuỗi AA sạch trước để biết AA tại mỗi vị trí
        clean_seq = re.sub(r'\(UniMod:\d+\)', '', peptide)
        nAA = len(clean_seq)
        
        # Danh sách kết quả
        mods = []
        mod_sites = []

        # 1. Xử lý N-term modification (đầu chuỗi)
        n_term_match = re.match(r'^\((UniMod:(\d+))\)', peptide)
        if n_term_match:
            uid = int(n_term_match.group(2))
            # Ưu tiên ánh xạ UniMod:1 ở đầu chuỗi về Any_N-term
            if uid == 1:
                mods.append("Acetyl@Any_N-term")
            else:
                # Tra cứu các loại N-term khác trong MOD_DF
                n_term_name = MOD_DF[(MOD_DF.unimod_id == uid) & 
                                    (MOD_DF.mod_name.str.contains('N-term'))].mod_name.iloc[0]
                mods.append(n_term_name)
            mod_sites.append(0)
            peptide = peptide[len(n_term_match.group(0)):]

        # 2. Xử lý các Modification trên Residue (giữa chuỗi)
        # Dùng regex để bắt cặp: AA đứng trước + (UniMod:XX)
        residue_matches = re.finditer(r'([A-Z])\((UniMod:(\d+))\)', peptide)
        
        # Để tính site chuẩn, ta dựa vào độ dài chuỗi sạch tính đến vị trí match
        for m in residue_matches:
            aa = m.group(1) # Acid amin (ví dụ: S, C, T)
            uid = int(m.group(3))
            
            # Tìm tên mod khớp cả ID và AA (ví dụ: Phospho@S thay vì Phospho@T)
            # Ta lọc MOD_DF theo unimod_id và cái đuôi '@AA'
            possible_names = MOD_DF[(MOD_DF.unimod_id == uid) & 
                                    (MOD_DF.mod_name.str.endswith(f"@{aa}"))].mod_name.values
            
            if len(possible_names) > 0:
                true_name = possible_names[0]
            else:
                # Nếu không tìm thấy @AA, lấy đại thằng đầu tiên của ID đó
                true_name = MOD_DF[MOD_DF.unimod_id == uid].mod_name.iloc[0]
                
            mods.append(true_name)
            
            # Tính vị trí thực tế trong clean_seq
            pre_str = peptide[:m.start(1)+1]
            site = len(re.sub(r'\(UniMod:\d+\)', '', pre_str))
            mod_sites.append(site)

        return clean_seq, ";".join(mods), ";".join(map(str, mod_sites)), nAA

    def _process_targets(self, df):
        frag_types = ["b_z1", "b_z2", "y_z1", "y_z2", "b_modloss_z1", "b_modloss_z2", "y_modloss_z1", "y_modloss_z2"]
        df = df.copy()
        df['charge_num'] = df['FragmentCharge'].astype(str).str.extract('(\d+)').astype(int)
        df['loss_clean'] = df['FragmentLossType'].apply(lambda x: '' if x in ['noloss', ''] else '_modloss')
        df['target_col'] = df['FragmentType'] + df['loss_clean'] + '_z' + df['charge_num'].astype(str)
        pivot = df.pivot_table(index=['psm_id', 'FragmentNumber'], columns='target_col', values='RelativeIntensity', aggfunc='max')
        pivot = pivot.reindex(columns=frag_types, fill_value=0.0).fillna(0)
        
        target_dict = {}
        for psm_id, group in pivot.groupby(level=0):
            padded = np.zeros((self.max_len, 8))
            mat = group.values
            padded[:min(len(mat), self.max_len), :] = mat[:self.max_len, :]
            target_dict[psm_id] = padded
        return target_dict

    def _pad_or_clip_2d(self, arr, max_len):
        res = np.zeros((arr.shape[0], max_len), dtype=arr.dtype)
        res[:, :min(arr.shape[1], max_len)] = arr[:, :min(arr.shape[1], max_len)]
        return res

    def _pad_or_clip_3d(self, arr, max_len):
        res = np.zeros((arr.shape[0], max_len, arr.shape[2]), dtype=arr.dtype)
        res[:, :min(arr.shape[1], max_len), :] = arr[:, :min(arr.shape[1], max_len), :]
        return res

    def __len__(self): return len(self.psm_ids)

    def __getitem__(self, idx):
        psm_id = self.psm_ids[idx]
        return {
            "aa_idx": torch.tensor(self.aa_indices[idx], dtype=torch.long),
            "mod_x": torch.tensor(self.mod_x[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[psm_id], dtype=torch.float32),
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
    
    # 2. Khởi tạo Dataset (max_len=30 để khớp Transformer)
    dataset = MS2Dataset(full_df, max_len=30)
    
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