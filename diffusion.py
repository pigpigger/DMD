# diffusion.py
import torch
import numpy as np

class Diffusion:
    def __init__(self, T, beta_start, beta_end, device):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)   # ᾱ_t

    def q_sample(self, x0, t, noise=None):
        """
        前向加噪：x_t = √ᾱ_t x0 + √(1-ᾱ_t) noise
        x0: (batch, dim)
        t:  (batch,) 整数时间步 (0-indexed)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)          # (batch,1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def get_alpha_sigma(self, t):
        """返回 α_t = √ᾱ_t, σ_t² = 1-ᾱ_t"""
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        alpha_t = torch.sqrt(alpha_bar_t)
        sigma_t_sq = 1 - alpha_bar_t
        return alpha_t, sigma_t_sq

    def score_from_pred(self, x_t, pred_x0, t):
        """
        根据预测的 x0 计算 score: s = -(x_t - α_t * pred_x0) / σ_t²
        """
        alpha_t, sigma_t_sq = self.get_alpha_sigma(t)
        score = -(x_t - alpha_t * pred_x0) / sigma_t_sq
        return score
