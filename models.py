# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_timestep_embedding


class ResidualBlock(nn.Module):
    """带残差连接的全连接块，可选归一化"""
    def __init__(self, in_dim, out_dim, activation=nn.SiLU, use_norm=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        # 兼容传入类或实例
        if isinstance(activation, type) and issubclass(activation, nn.Module):
            self.act = activation()
        else:
            self.act = activation
        self.use_residual = (in_dim == out_dim)
        if not self.use_residual and in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        """前向传播，包含残差连接"""
        identity = self.skip(x)
        out = self.fc(x)
        out = self.norm(out)
        out = self.act(out)
        out = out + identity
        return out

class MLP(nn.Module):
    """改进的去噪器：预测 x₀，带时间嵌入和残差块"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3, time_embed_dim=128):
        super().__init__()
        # 时间嵌入网络（输出 hidden_dim）
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 输入拼接层（将 x 和 t_emb 合并）
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # 残差块序列
        blocks = []
        for i in range(num_layers - 2):
            blocks.append(ResidualBlock(hidden_dim, hidden_dim, activation=nn.SiLU, use_norm=True))
        self.blocks = nn.Sequential(*blocks)
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        # 时间嵌入
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)  # (batch, time_embed_dim)
        t_emb = self.time_embed(t_emb)                                     # (batch, hidden_dim)
        # 拼接输入
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_proj(h)              # 投影到 hidden_dim
        h = self.blocks(h)                  # 残差块
        out = self.output_proj(h)           # 输出 x0 预测
        return out

class Generator(nn.Module):
    """改进的生成器：从噪声 z 映射到数据 x，使用残差块"""
    def __init__(self, z_dim, output_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        blocks = []
        in_dim = z_dim
        # 第一层投影
        blocks.append(nn.Linear(in_dim, hidden_dim))
        blocks.append(nn.SiLU())
        # 中间残差块
        for _ in range(num_layers - 2):
            blocks.append(ResidualBlock(hidden_dim, hidden_dim, activation=nn.SiLU, use_norm=True))
        # 输出层
        blocks.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """改进的判别器：带时间嵌入和残差块，输出 logit"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, time_embed_dim=128, use_spectral_norm=False):
        super().__init__()
        # 时间嵌入网络
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 输入投影
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # 残差块序列
        blocks = []
        for i in range(num_layers - 2):
            blocks.append(ResidualBlock(hidden_dim, hidden_dim, activation=nn.LeakyReLU(0.2), use_norm=True))
        self.blocks = nn.Sequential(*blocks)
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)

        # 可选谱归一化（稳定 GAN 训练）
        if use_spectral_norm:
            self.apply_spectral_norm()

    def apply_spectral_norm(self):
        for module in [self.input_proj, self.output_layer] + list(self.blocks.modules()):
            if isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module)

    def forward(self, x, t):
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_proj(h)
        h = self.blocks(h)
        out = self.output_layer(h).squeeze(-1)
        return out
