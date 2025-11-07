import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# ======================
# Baseline Transformer
# ======================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embd
        qkv = self.qkv_proj(x)  # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)  # mỗi cái [B, T, C]

        # chia thành nhiều đầu
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nH, T, dH]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nH, T, T]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ v  # [B, nH, T, dH]

        # gộp các head lại
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
@dataclass
class Config:    
    block_size: int = None
    vocab_size: int = None
    n_layers: int = 2
    n_heads: int = 8
    n_embd: int = 512
    d_ff: int = 1024
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    CLS: int = None
    device: str = None

class TransformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config.n_embd, config.d_ff, config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(config)
        for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.predict_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd//2),
            nn.GELU(),
            nn.Linear(config.n_embd//2, 1)
        )

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        cls_token = x[:, 0, :]  
        out = self.predict_head(cls_token)
        return out

 
