# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
from nncore.nn import (MODELS, FeedForwardNetwork, MultiHeadAttention,
                       Parameter, build_norm_layer)
from nncore.nn.blocks.transformer import TransformerEncoderLayer
from mamba_ssm import Mamba


@MODELS.register()
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to("cuda")
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)

@MODELS.register()
class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim=None, output_dim=None):
    super().__init__()

    self.input_dim = input_dim
    self.hidden_dim = input_dim*2 if hidden_dim is None else hidden_dim
    self.output_dim = input_dim if output_dim is None else output_dim
    # 输入到隐藏层
    self.fc1 = nn.Linear(self.input_dim, self.hidden_dim).to("cuda")
    # 隐藏层
    self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to("cuda")
    # 隐藏层到输出层
    self.fc3 = nn.Linear(self.hidden_dim, self.output_dim).to("cuda")
    
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    
    return x

@MODELS.register()
class mamba_encoder(nn.Module):
    
    def __init__(self, dims=None, pos_cfg=None, dec_cfg=None, norm_cfg=None):
        super(mamba_encoder, self).__init__()

        self.rmsnorm_1 = LlamaRMSNorm(hidden_size=dims)
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dims, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            ).to("cuda")
        self.rmsnorm_2 = LlamaRMSNorm(hidden_size=dims)
        self.mlp = MLP(input_dim=dims)

    def forward(self, x):
        x_0 = self.rmsnorm_1(x)
        x_1 = self.mamba(x_0)
        x = x_1 + x
        x_2 = self.rmsnorm_2(x)
        x_3 = self.mlp(x_2)
        x = x + x_3

        return x


@MODELS.register()
class CoAttentionModule(nn.Module):
    def __init__(self, dim):
        super(CoAttentionModule, self).__init__()
        self.dim = dim // 2
        self.h_projection = nn.Linear(self.dim, 1)

    def forward(self, modal1, modal2):
        # Project modalities to a common dimension
        # f = self.projection(modal1)  # Shape: (batch, seq_len1, dim)
        # g = self.projection(modal2)  # Shape: (batch, seq_len2, dim)

        # Expand modalities to calculate attention
        f_expand = modal1.unsqueeze(2)  # Shape: (batch, seq_len1, 1, dim)
        g_expand = modal2.unsqueeze(1)  # Shape: (batch, 1, seq_len2, dim)

        # Compute compatibility score
        fg = torch.tanh(f_expand + g_expand)  # Broadcasting
        attention = self.h_projection(fg).squeeze(-1)  # Shape: (batch, seq_len1, seq_len2)

        # Compute attention weights
        attention1 = F.softmax(attention, dim=2)  # Shape: (batch, seq_len1, seq_len2)
        attention2 = F.softmax(attention, dim=1)  # Shape: (batch, seq_len1, seq_len2)

        # Apply attention weights
        co_feature1 = torch.matmul(attention1, modal2)  # Shape: (batch, seq_len1, dim)
        co_feature2 = torch.matmul(attention2.transpose(1, 2), modal1)  # Shape: (batch, seq_len2, dim)

        return co_feature1, co_feature2

@MODELS.register()
class BottleneckTransformerLayer(nn.Module):

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(BottleneckTransformerLayer, self).__init__()

        self.dims = dims
        self.heads = heads
        self.ratio = ratio
        self.p = p

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att3 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att4 = MultiHeadAttention(dims, heads=heads, p=p)

        self.ffn1 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)
        self.ffn2 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)
        self.norm4 = build_norm_layer(norm_cfg, dims=dims)
        self.norm5 = build_norm_layer(norm_cfg, dims=dims)
        self.norm6 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, a, b, t, pe=None, mask=None):

        da = self.norm1(a)
        db = self.norm2(b)
        dt = self.norm3(t)

        ka = da if pe is None else da + pe
        kb = db if pe is None else db + pe

        at = self.att1(dt, ka, da, mask=mask)
        bt = self.att2(dt, kb, db, mask=mask)


        t = t + at + bt

        dt = self.norm4(t)


        qa = da if pe is None else da + pe
        qb = db if pe is None else db + pe

        a = a + self.att3(qa, dt)
        b = b + self.att4(qb, dt)

        da = self.norm5(a)
        db = self.norm6(b)

        a = a + self.ffn1(da)
        b = b + self.ffn2(db)

        return a, b, t


@MODELS.register()
class BottleneckTransformer(nn.Module):

    def __init__(self, dims, num_tokens=4, num_layers=3, **kwargs):
        super(BottleneckTransformer, self).__init__()

        # dims = dims*3
        self.dims = dims
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        # self.token = Parameter(num_tokens, dims)


        
        # self.mapping = nn.Linear(dims*2, dims)
        self.mapping = nn.Linear(2816, 512)


        # self.encoder_1 = nn.ModuleList([
        #     CoAttentionModule(dims)
        #     for _ in range(num_layers)
        # ])
        # self.encoder_1 = nn.ModuleList([
        #     TransformerEncoderLayer(dims, **kwargs)
        #     for _ in range(num_layers)
        # ])

        # self.encoder_2 = nn.ModuleList([
        #     BottleneckTransformerLayer(dims, **kwargs)
        #     for _ in range(num_layers)
        # ])

        # self.encoder_1 = nn.ModuleList([

        #     # mamba_encoder(dims=dims).to("cuda")

        #     Mamba(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=dims, # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,    # Local convolution width
        #     expand=2,    # Block expansion factor
        #     ).to("cuda")
            
        #     for _ in range(num_layers)
        # ])
        

    # def forward(self, a, b, **kwargs):
    #     t = self.token.expand(a.size(0), -1, -1)
    #     for enc in self.encoder:
    #         a, b, t = enc(a, b, t, **kwargs)
    #     return a, b
    
    def forward(self, a, b=None, c=None,**kwargs):

        # t = self.token.expand(a.size(0), -1, -1)
        # for enc in self.encoder_2:
        #     a, b, t = enc(a, b, t, **kwargs)
        # return a, b

        # for enc in self.encoder_1:
        #     d = enc(a)
        
        # b = a + b
        b = torch.cat((a, b), dim=-1)
        b = self.mapping(b)

       
        
        return d


