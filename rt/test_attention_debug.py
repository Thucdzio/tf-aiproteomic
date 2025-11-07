import torch
from model import Config, MultiHeadSelfAttention, SimpleTransformer
import torch.optim as optim
import torch.nn as nn
from dataloader import DatasetRT
import matplotlib.pyplot as plt

# --- Load 1 batch dữ liệu thật ---
data = DatasetRT(dataset='deepdia', batch_size=32, epochs=1, device='cuda' if torch.cuda.is_available() else 'cpu')
x, y = data.get_batch('train')

print(x)
print("Input batch shape:", x.shape)  # [B, T]
print("Example:", x[0][:10])  # in thử 10 token đầu tiên

# --- Cấu hình model ---
config = Config(
    n_embd=64,       # embedding size nhỏ để debug nhanh
    n_heads=4,        # 4 head
    d_ff=128,         # feedforward hidden dim
    dropout=0.0,
    bias=False,
    block_size=data.block_size,
    vocab_size=data.vocab_size,
    CLS=data.CLS,
    device=data.device
)
# --- Tạo model, optimizer, loss function ---
model = SimpleTransformer(config=config)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()

for step in range(200):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss {loss.item():.4f}")

print("Predictions:", y_pred)
print("Targets:", y) 
plt.scatter(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
plt.show()