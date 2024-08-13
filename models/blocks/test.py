import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.learnable = learnable
        if learnable:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, base=10000, learnable=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.rotary_embedding = RotaryEmbedding(self.head_dim, base, learnable=learnable)

    def forward(self, x):
        cos, sin = self.rotary_embedding(x)
        x = self.apply_rotary_emb(x, cos, sin)
        for layer in self.layers:
            x = layer(x)
        return x

    def apply_rotary_emb(self, x, cos, sin):
        bsz, seq_len, dim = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        
        cos = cos[:, None, None, :].expand(-1, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        sin = sin[:, None, None, :].expand(-1, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        
        x1, x2 = x.chunk(2, dim=-1)
        rotary_emb = torch.cat([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1)
        
        return rotary_emb.view(bsz, seq_len, dim)

# 示例使用
dim = 512
num_heads = 8
num_layers = 6
seq_len = 100
batch_size = 32

model = SimpleTransformerEncoder(dim, num_heads, num_layers)
input_data = torch.randn(batch_size, seq_len, dim)
output = model(input_data)
print(output.shape)  # 输出: torch.Size([32, 100, 512])
