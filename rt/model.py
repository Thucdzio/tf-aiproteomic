import torch
import torch.nn as nn
from torch.nn import functional as F
import math

batch_size = 16       # Batch size
block_size = 51       # Độ dài chuỗi (50 peptide + 1 CLS)
max_iters = 1000      # Số vòng lặp train
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128          # Kích thước vector (nhỏ lại chút cho nhẹ)
n_head = 4
n_layer = 4
dropout = 0.1
protein_size = 30       # 20 axit amin + các ký tự đặc biệt + CLS

print(f"Using device: {device}")

class DummyDataLoader:
    def __init__(self):
        print("Đang tạo dữ liệu giả lập để test...")
        
    def next_batch(self, split):
    
        x = torch.randint(1, 20, (batch_size, block_size))
        x[:, 0] = 21 # CLS token
        x[:, -5:] = 0 # Giả vờ padding 5 số cuối
        
        # Tạo Y giả: [Batch, 1]
        # Giá trị RT từ 0 đến 1
        y = torch.rand((batch_size, 1))
        
        return x.to(device), y.to(device)


class Head(nn.Module):
    """ Một đầu Attention (Self-attention head) """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # [QUAN TRỌNG]: Đã xóa dòng register_buffer('tril'...) 
        # Vì Encoder cần nhìn thấy tương lai, không cần che mặt nạ.

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        # Tính attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        
        # [QUAN TRỌNG]: Đã xóa dòng masked_fill 
        # Để các token nhìn thấy nhau toàn bộ (Full Attention)
        
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ Nhiều đầu Attention chạy song song """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ Mạng nơ-ron truyền thẳng đơn giản """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Một khối Transformer Block """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ProteinRTModel(nn.Module):
    """ Model chính: Encoder cho dự đoán RT """

    def __init__(self):
        super().__init__()
        # Embedding cho Axit Amin
        self.token_embedding_table = nn.Embedding(protein_size, n_embd)
        # Embedding cho vị trí (Positional Encoding)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Chồng các Block lên nhau (Encoder)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        
        # [QUAN TRỌNG]: REGRESSION HEAD
        # Đầu ra là 1 số thực (RT)
        self.rt_head = nn.Sequential(
            nn.Linear(n_embd, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. Tạo Embedding
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb 
        
        # 2. Qua Encoder
        x = self.blocks(x) 
        x = self.ln_f(x) # (B,T,C)

      
        cls_vector = x[:, 0, :] # Shape: (B, C)

        # 4. Dự đoán RT
        rt_pred = self.rt_head(cls_vector) # Shape: (B, 1)

        loss = None
        if targets is not None:
            # [QUAN TRỌNG]: Hàm Loss là MSE (Mean Squared Error) cho Regression
            loss = F.mse_loss(rt_pred, targets)

        return rt_pred, loss

# Khởi tạo model
model = ProteinRTModel()
m = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Khởi tạo DataLoader (Dùng cái giả lập)
loader = DummyDataLoader()

print("\nBắt đầu train...")
for iter in range(max_iters):

    # 1. Lấy dữ liệu
    xb, yb = loader.next_batch('train')

    # 2. Tính toán (Forward)
    preds, loss = model(xb, yb)

    # 3. Cập nhật (Backward)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 4. In thông báo
    if iter % 100 == 0:
        print(f"Step {iter}: Loss = {loss.item():.6f}")

print("\nTrain xong! Thử dự đoán 1 mẫu:")
with torch.no_grad():
    sample_x, sample_y = loader.next_batch('test')
    pred, _ = model(sample_x, sample_y)
    print(f"Input shape: {sample_x.shape}")
    print(f"Target RT: {sample_y.item():.4f}")
    print(f"Predicted RT: {pred.item():.4f}")