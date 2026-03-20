# models.py
import torch
import torch.nn as nn
from utils import get_timestep_embedding

class MLP(nn.Module):
    """简单的 MLP，用于去噪器（预测 x₀）或生成器"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3, time_embed_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 输入为 x (input_dim) 拼接时间嵌入 (hidden_dim)
        layers = []
        in_dim = input_dim + hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # t: (batch,) 整数时间步
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)  # (batch, time_embed_dim)
        t_emb = self.time_embed(t_emb)                                     # (batch, hidden_dim)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)

class Generator(nn.Module):
    """生成器：从噪声 z 映射到数据 x"""
    def __init__(self, z_dim, output_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, time_embed_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        layers = []
        in_dim = input_dim + hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # 输出单个 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h).squeeze(-1)  # 返回 (batch,)
