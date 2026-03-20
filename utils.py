# utils.py
import torch
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    正弦时间嵌入，与 Transformer 类似
    timesteps: (batch,) 整数时间步
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def visualize_2d(samples, path='samples.png'):
    """简单的 2D 散点图可视化"""
    import matplotlib.pyplot as plt
    
    # 显式创建 figure 和 axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 在 axes 上绘图
    ax.scatter(samples[:,0].cpu(), samples[:,1].cpu(), 
               s=2, alpha=0.5, c='green')
    ax.set_title('Samples')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, alpha=0.3)
    
    # 保存并关闭
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)  # 明确关闭这个 figure

