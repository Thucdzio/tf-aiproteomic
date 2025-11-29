import torch
import numpy as np
from data_loader import DataLoaderRT, data_generics  # Import class từ file của mày

# 1. Cấu hình
DATASET_NAME = 'deepdia'  # Hoặc 'prosit', 'autort', 'phospho'
BATCH_SIZE = 4            # Lấy thử 4 mẫu thôi cho dễ nhìn

# 2. Khởi tạo DataLoader
# Lưu ý: Code của mày yêu cầu phải có file .npy trong thư mục ./data
print(f"Dang load du lieu {DATASET_NAME}...")
loader = DataLoaderRT(dataset=DATASET_NAME, batch_size=BATCH_SIZE)

# 3. Lấy 1 batch dữ liệu ra xem
# Hàm next_batch trả về (x, y)

# x: Tensor chứa peptide đã mã hóa
# y: Tensor chứa giá trị RT (đã scale 0-1)
batch_x, batch_y = loader.next_batch('train')

# 4. In thông tin kiểm tra
print("\n" + "="*30)
print("KIỂM TRA INPUT (X)")
print("="*30)
print(f"Shape của X: {batch_x.shape}") 
# Dự kiến: (4, 51) -> 4 dòng, 50 token peptide + 1 token CLS

print("\n--- Mẫu đầu tiên (Dạng số/Tensor) ---")
print(batch_x[0]) 
# Mày sẽ thấy số đầu tiên là CLS (ví dụ 22), sau đó là các số 1, 2, 5... và cuối cùng là 0 (padding)

# Dịch ngược lại từ số sang chữ để xem peptide là gì
# Lưu ý: Phải bỏ cột đầu tiên (CLS token) đi thì mới dịch đúng dictionary được
peptide_ints = batch_x[0, 1:].numpy() 
peptide_str = data_generics.integer_to_sequence([peptide_ints])[0]
print(f"\n--- Dịch sang chuỗi Peptide: {peptide_str}")

print("\n" + "="*30)
print("KIỂM TRA OUTPUT (Y)")
print("="*30)
print(f"Shape của Y: {batch_y.shape}")
# Dự kiến: (4, ) hoặc (4, 1)

print("\n--- Giá trị RT (Scaled 0-1) ---")
print(batch_y)

# Thử un-scale về giá trị thực (phút) xem có hợp lý không
# Hàm min_max_scale_rev: x * (max - min) + min
real_rt = batch_y * (loader.max_val - loader.min_val) + loader.min_val
print("\n--- Giá trị RT Thực tế (Phút) ---")
print(real_rt)