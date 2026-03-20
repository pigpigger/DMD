# data.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import math
def generate_complex_data(n_samples, mode='spiral', device='cpu', **kwargs):
    """
    生成各种复杂分布的数据。
    
    参数:
        n_samples: 总样本数
        mode: 生成模式，可选:
            - 'rotated_gmm': 旋转混合高斯（默认）
            - 'ring': 环形分布
            - 'figure8': 八字形（两个环相交）
            - 'spiral': 螺旋形
            - 'grid': 网格状高斯混合
        device: 设备
        **kwargs: 模式相关参数
    返回:
        data: shape (n_samples, 2)
    """
    if mode == 'rotated_gmm':
        # 旋转混合高斯（各向异性）
        # 可选参数: means (list of [x,y]), scales (list of [sx,sy]), angles (list of angle in degrees)
        means = kwargs.get('means', [[-4, -4], [4, 4], [0, 0]])
        scales = kwargs.get('scales', [[1.5, 0.5], [1.5, 0.5], [0.8, 0.8]])
        angles = kwargs.get('angles', [45, -30, 0])  # 旋转角度（度）
        weights = kwargs.get('weights', None)        # 每个成分的权重，None表示均匀
        
        num_components = len(means)
        if weights is None:
            weights = [1.0/num_components] * num_components
        # 根据权重分配样本数
        comp_samples = [int(n_samples * w) for w in weights]
        comp_samples[-1] = n_samples - sum(comp_samples[:-1])
        
        data_list = []
        for i in range(num_components):
            cnt = comp_samples[i]
            if cnt <= 0:
                continue
            # 生成标准正态样本
            z = torch.randn(cnt, 2, device=device)
            # 缩放
            sx, sy = scales[i]
            z[:, 0] *= sx
            z[:, 1] *= sy
            # 旋转
            rad = math.radians(angles[i])
            cos = math.cos(rad)
            sin = math.sin(rad)
            rot = torch.tensor([[cos, -sin], [sin, cos]], device=device, dtype=torch.float32)
            z = z @ rot.T
            # 平移
            z += torch.tensor(means[i], device=device)
            data_list.append(z)
        data = torch.cat(data_list, dim=0)
    
    elif mode == 'ring':
        # 环形分布
        radius = kwargs.get('radius', 5.0)
        thickness = kwargs.get('thickness', 0.5)
        theta = torch.rand(n_samples, device=device) * 2 * math.pi
        r = radius + torch.randn(n_samples, device=device) * thickness
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        data = torch.stack([x, y], dim=1)
    
    elif mode == 'figure8':
        # 八字形（两个环相交，参数方程）
        t = torch.rand(n_samples, device=device) * 2 * math.pi
        scale = kwargs.get('scale', 3.0)
        x = scale * torch.sin(t)
        y = scale * torch.sin(t) * torch.cos(t)
        # 加噪声
        noise = torch.randn(n_samples, 2, device=device) * 0.2
        data = torch.stack([x, y], dim=1) + noise
    
    elif mode == 'spiral':
        # 螺旋形
        t = torch.rand(n_samples, device=device) * 6 * math.pi
        a = kwargs.get('a', 1.0)
        b = kwargs.get('b', 0.3)
        r = a + b * t
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        # 加径向噪声
        noise = torch.randn(n_samples, 2, device=device) * 0.2
        data = torch.stack([x, y], dim=1) + noise
    
    elif mode == 'grid':
        # 网格状高斯混合（2x2 网格）
        spacing = kwargs.get('spacing', 5.0)
        offset = kwargs.get('offset', 0.0)
        centers = []
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                centers.append([i*spacing/2 + offset, j*spacing/2 + offset])
        data = generate_complex_data(n_samples, mode='rotated_gmm', device=device,
                                     means=centers, scales=[[0.8,0.8]]*4, angles=[0]*4)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return data
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
    #data = generate_mixture_of_gaussians(n_samples, dim, num_components, device)
    data = generate_complex_data(n_samples)
    
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
