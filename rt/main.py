import numpy as np
import pandas as pd

# # Đọc file .npy
# data = np.load('data/PXD005573_rt_x_train.npy')

# # Xem nội dung
# print(data)
# print(type(data))
# print(data.shape)

df = pd.read_csv('data/PXD019038-hct116.csv')

# Xem dữ liệu
print(df.head())       # In 5 dòng đầu
print(df.info())  