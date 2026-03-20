# data.py
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_mixture_of_gaussians(n_samples, dim=2, num_components=2, device='cpu'):
    """
    生成混合高斯数据，每个成分权重相等，随机均值和单位协方差
    """
    #means = torch.randn(num_components, dim) * 3   # 各中心随机
    means = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], device='cpu')
    # 每个样本随机选择一个成分
    component = torch.randint(0, num_components, (n_samples,))
    samples = []
    for i in range(num_components):
        mask = component == i
        cnt = mask.sum().item()
        if cnt > 0:
            # 生成标准高斯样本
            x = torch.randn(cnt, dim)
            # 计算半径
            r = torch.norm(x, dim=1, keepdim=True)
            # 各向同性幂变换：新半径 = r^p，保持方向
            # x_new = x * (r^(p-1) / r) = x * r^(p-2)
            tail_power=3
            x = x * (r ** (tail_power - 1)) / (r + 1e-8)
            samples.append(x + means[i]) 
    data = torch.cat(samples, dim=0)
    return data
def get_dataloader(batch_size, n_samples, dim=2, num_components=2, device='cpu'):
    """
    生成混合高斯数据，并保存数据可视化图片到 real.png
    """
    data = generate_mixture_of_gaussians(n_samples, dim, num_components, device)
    
    # 如果是2D数据，保存可视化图片
    if dim == 2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), s=2, alpha=0.5, c='blue')
        plt.title(f'Real Data (n={n_samples})')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('real.png', dpi=100)
        plt.close()
        print(f"真实数据可视化已保存到 real.png")
    
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
