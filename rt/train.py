import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleTransformer
# dữ liệu giả
x = torch.randint(0, 25, (32, 10))
y = torch.rand(32, 1)

model = SimpleTransformer(vocab_size=25, max_len=10)
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
