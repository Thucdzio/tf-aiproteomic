import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader import MS2Dataset, LengthGroupedSampler, collate_fn_ms2_grouped
from model.ms2_model import ModelMS2Transformer # Đảm bảo đúng đường dẫn
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ==== 1. Cấu hình ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001
EPOCHS = 1000
BATCH_SIZE = 32 
SAVE_PATH = "./checkpoint"
os.makedirs(SAVE_PATH, exist_ok=True)

def calculate_metrics(target, pred):
    """
    Tính PCC và SA cho toàn batch. 
    Target và Pred shape: (Batch, L-1, 8)
    """
    # Flatten L và 8 thành 1 vector duy nhất cho mỗi peptide trong batch
    # Shape: (Batch, (L-1)*8)
    t = target.flatten(start_dim=1)
    p = pred.flatten(start_dim=1)

    # 1. Pearson Correlation Coefficient (PCC)
    t_mean = torch.mean(t, dim=1, keepdim=True)
    p_mean = torch.mean(p, dim=1, keepdim=True)
    t_centered = t - t_mean
    p_centered = p - p_mean
    
    num = torch.sum(t_centered * p_centered, dim=1)
    den = torch.sqrt(torch.sum(t_centered**2, dim=1) * torch.sum(p_centered**2, dim=1) + 1e-8)
    pcc = torch.mean(num / den)

    # 2. Spectral Angle (SA)
    # SA = 1 - 2*arccos( (t.p) / (||t||.||p||) ) / pi
    # Ở đây dùng phiên bản đơn giản thường dùng trong Proteomics:
    dot_product = torch.sum(t * p, dim=1)
    t_norm = torch.norm(t, p=2, dim=1)
    p_norm = torch.norm(p, p=2, dim=1)
    
    # Kẹp giá trị trong khoảng [-1, 1] để tránh lỗi arccos
    cos_sim = torch.clamp(dot_product / (t_norm * p_norm + 1e-8), -1.0, 1.0)
    spectral_angle = 1 - (2 * torch.acos(cos_sim)) / np.pi
    sa = torch.mean(spectral_angle)

    return pcc.item(), sa.item()


def train():

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
    unique_psms = full_df['psm_id'].unique()
    train_psms, test_psms = train_test_split(unique_psms, test_size=0.1, random_state=42)

    # 2. Tạo dataframe riêng cho Train và Test
    train_df = full_df[full_df['psm_id'].isin(train_psms)].copy()
    test_df = full_df[full_df['psm_id'].isin(test_psms)].copy()

    # 3. Khởi tạo 2 Dataset riêng biệt
    train_dataset = MS2Dataset(train_df)
    test_dataset = MS2Dataset(test_df)

    # 4. Tạo Sampler và DataLoader cho Train
    train_sampler = LengthGroupedSampler(train_dataset.nAA_list, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn_ms2_grouped)

    # 5. Với Test: Không cần shuffle, bốc lần lượt để đánh giá
    test_sampler = LengthGroupedSampler(test_dataset.nAA_list, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn_ms2_grouped)


    
    # ==== 3. Khởi tạo Model ====
    model = ModelMS2Transformer(
        num_frag_types=8,
        num_modloss_types=4,
        hidden=64,
        nlayers=2,
        mask_modloss=False,
        dropout=0.3
    ).to(device)

    print(f"Tham sô model: {sum(p.numel() for p in model.parameters())}")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # L1Loss như alphapeepdep
    criterion = nn.L1Loss() 

    print(f"Bắt đầu Train trên: {device}")

    for epoch in range(EPOCHS):
        # --- PHASE 1: TRAINING ---
        model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            # Đẩy dữ liệu lên GPU
            aa_idx = batch['aa_idx'].to(device)
            mod_x = batch['mod_x'].to(device)
            charge = batch['charge'].to(device)
            nce = batch['nce'].to(device)
            inst = batch['instrument'].to(device)
            targets = batch['target'].to(device)

            # Forward
            preds = model(aa_idx, mod_x, charge, nce, inst)

            # Tính Loss
            loss = criterion(preds, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # --- PHASE 2: EVALUATION (TEST) ---
        model.eval()
        test_pcc = 0
        test_sa = 0
        with torch.no_grad():
            for batch in test_loader:
                aa = batch['aa_idx'].to(device)
                mod = batch['mod_x'].to(device)
                charge = batch['charge'].to(device)
                nce = batch['nce'].to(device)
                inst = batch['instrument'].to(device)
                targets = batch['target'].to(device)

                preds = model(aa, mod, charge, nce, inst)
                
                pcc, sa = calculate_metrics(targets, preds)
                test_pcc += pcc
                test_sa += sa

        avg_train_loss = train_loss / len(train_loader)
        avg_test_pcc = test_pcc / len(test_loader)
        avg_test_sa = test_sa / len(test_loader)

        with open(os.path.join(SAVE_PATH, "train_log.txt"), "a") as f:
            f.write(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Test PCC: {avg_test_pcc:.4f} - Test SA: {avg_test_sa:.4f}\n")
        if ( epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Test PCC: {avg_test_pcc:.4f} - Test SA: {avg_test_sa:.4f}")

if __name__ == "__main__":
    train()