import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Import DataLoader từ file của mày
# Nếu mày dùng dummy thì import DummyDataLoader, dùng thật thì import DataLoaderRT
try:
    from data_loader import DataLoaderRT as Loader
    DATASET_NAME = 'deepdia' # Hoặc 'prosit', 'autort'
except ImportError:
    from train_rt_encoder import DummyDataLoader as Loader
    DATASET_NAME = 'dummy'

def get_vocab_map():
    # Mapping ngược từ số sang chữ (dựa trên code data_generics của mày)
    # 0: Pad, 21/22: CLS, còn lại là Axit amin
    # Đây là map ví dụ chuẩn, nếu map của mày khác thì chỉnh lại chút
    int2seq = {
        1:'A', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L',
        11:'M', 12:'N', 13:'P', 14:'Q', 15:'R', 16:'S', 17:'T', 18:'V', 19:'W', 20:'Y',
        21:'[CLS]', 0:'_'
    }
    return int2seq

def decode_seq(tensor_seq, int2seq):
    # tensor_seq: [21, 1, 5, ..., 0, 0]
    seq = []
    for token in tensor_seq:
        idx = token.item()
        if idx == 0: continue # Bỏ padding
        if idx in [21, 22]: continue # Bỏ CLS (hoặc giữ tùy mày)
        char = int2seq.get(idx, '?')
        seq.append(char)
    return "".join(seq)

def visualize():
    print(f"--- Đang tải dữ liệu ({DATASET_NAME})... ---")
    
    # 1. Load 1 Batch dữ liệu
    if DATASET_NAME == 'dummy':
        loader = Loader()
    else:
        loader = Loader(dataset=DATASET_NAME, batch_size=64)
        
    x_batch, y_batch = loader.next_batch('train')
    
    # Chuyển sang CPU / Numpy để vẽ
    x_np = x_batch.cpu().numpy()
    y_np = y_batch.cpu().numpy().flatten() # Flatten thành mảng 1 chiều
    
    # ==========================================
    # HÌNH 1: BIỂU ĐỒ PHÂN BỐ RT (DISTRIBUTION)
    # ==========================================
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    # Vẽ Histogram
    sns.histplot(y_np, bins=20, kde=True, color='skyblue', edgecolor='black')
    
    plt.title(f'Phân bố giá trị RT trong một Batch (Dataset: {DATASET_NAME})', fontsize=14)
    plt.xlabel('Giá trị RT (Normalized [0, 1])', fontsize=12)
    plt.ylabel('Số lượng Peptide', fontsize=12)
    plt.axvline(x=y_np.mean(), color='red', linestyle='--', label=f'Mean: {y_np.mean():.2f}')
    plt.legend()
    
    # Lưu ảnh
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print(">> Đã lưu ảnh: data_distribution.png")
    plt.close()

    # ==========================================
    # HÌNH 2: BẢNG DỮ LIỆU MINH HỌA (DATAFRAME)
    # ==========================================
    # Lấy 10 mẫu đầu tiên để hiển thị
    num_samples = 5
    int2seq = get_vocab_map()
    
    data_list = []
    for i in range(num_samples):
        # Decode chuỗi số thành chuỗi ký tự
        seq_str = decode_seq(x_batch[i], int2seq)
        
        # Lấy token raw (cắt bớt padding cho gọn để hiển thị)
        raw_tokens = str(x_np[i][:15]) + "..." # Chỉ lấy 15 số đầu minh họa
        
        rt_val = f"{y_np[i]:.4f}"
        
        data_list.append({
            # "Index": i,
            "Raw Tokens (Input)": raw_tokens,
            "Decoded Sequence": seq_str,
            
        })
        
    df = pd.DataFrame(data_list)
    
    # Vẽ bảng thành ảnh
    fig, ax = plt.subplots(figsize=(12, 4)) # Kích thước bảng
    ax.axis('tight')
    ax.axis('off')
    
    # Tạo bảng
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    # Chỉnh font size và độ rộng cột
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8) # Zoom bảng lên cho dễ đọc
    
    # Tô màu tiêu đề cột
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
    
    
    plt.savefig('data_samples_table.png', dpi=300, bbox_inches='tight')
    print(">> Đã lưu ảnh: data_samples_table.png")
    plt.close()

if __name__ == "__main__":
    visualize()