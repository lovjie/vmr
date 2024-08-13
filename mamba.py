import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# assert y.shape == x.shape
# print(y.shape)
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


class mamba_encoder(nn.Module):
    
    def __init__(self, dims=None, pos_cfg=None, dec_cfg=None, norm_cfg=None):
        super(mamba_encoder, self).__init__()

        self.rmsnorm_1 = LlamaRMSNorm()
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            ).to("cuda")
        self.rmsnorm_2 = LlamaRMSNorm()
        self.mlp = MLP(input_dim=16)

    def forward(self, x):
        x_0 = self.rmsnorm_1(x)
        x_1 = self.mamba(x_0)
        x = x_1 + x
        x_2 = self.rmsnorm_2(x)
        x_3 = self.mlp(x_2)
        x = x + x_3

        return x

model = mamba_encoder()
y = model(x)
print(y.shape)

