# train_dmd.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这一行
import torch.optim as optim
from models import Generator, MLP
from diffusion import Diffusion
from data import get_dataloader  # 用于生成真实样本（训练 fake score 时也需要参考？实际上 fake score 只用生成的假样本）
import os
#from utils import visualize_2d
import matplotlib.pyplot as plt
def generate_paired_data(teacher, diffusion, cfg, num_pairs=10000):
    """
    预计算 noise-output pairs 用于回归损失
    """
    print(f"Generating {num_pairs} paired data for regression loss...")
    paired_z = []
    paired_y = []
    
    teacher.eval()
    with torch.no_grad():
        for i in range(0, num_pairs, cfg.batch_size):
            batch_sz = min(cfg.batch_size, num_pairs - i)
            
            # 1. 生成随机噪声 z，并用它做 teacher 目标 x_T 的起点
            z = torch.randn(batch_sz, cfg.data_dim, device=cfg.device)
            x = z.clone()
            alpha_bars = diffusion.alpha_bars
            
            # 快速采样（50步）
            #steps = 50
            #step_indices = torch.linspace(cfg.T-1, 0, steps+1, dtype=torch.long, device=cfg.device)
            
            alpha_bars = diffusion.alpha_bars
            #for j in range(len(step_indices)-1):
            #    t = step_indices[j]
            #    t_tensor = torch.full((batch_sz,), t, device=cfg.device, dtype=torch.long)
            #    pred_x0 = teacher(x, t_tensor)
            #    
            #    alpha_bar_t = alpha_bars[t]
            #    alpha_bar_next = alpha_bars[step_indices[j+1]]
            #    
            #    sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
                # 这里使用 DDIM 风格（eta=0）保持 z->y 的确定性映射，防止 paired 样本一一对应被噪声破坏
            #    x = sqrt_alpha_bar_next * pred_x0 + torch.sqrt(1 - alpha_bar_next) * torch.zeros_like(x)
           
            for t in range(cfg.T - 1, 0, -1):
                t_tensor = torch.full((x.size(0),), t, device=cfg.device, dtype=torch.long)

                # 获取当前步和上一步的 alpha_bar
                alpha_bar_t = alpha_bars[t]
                alpha_bar_t_prev = alpha_bars[t-1]

                # 预计算平方根项，加 1e-8 防止数值误差导致负数
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t + 1e-8)
                sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev + 1e-8)

                # 1. 模型预测 x0
                pred_x0 = teacher(x, t_tensor)

                # 2. 计算噪声方向 (direction pointing to x_t)
                # 公式: epsilon_theta = (x_t - sqrt(alpha_bar_t) * pred_x0) / sqrt(1 - alpha_bar_t)
                # 这一步本质是在估算当前的噪声成分
                dir_xt = (x - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t

                # 3. 计算 x_{t-1} (确定性更新, sigma_t = 0)
                # 公式: x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1}) * dir_xt
                x = sqrt_alpha_bar_t_prev * pred_x0 + sqrt_one_minus_alpha_bar_t_prev * dir_xt


            paired_z.append(z.cpu())
            paired_y.append(x.cpu())
           
            
            if (i+1) % 2000 == 0:
                print(f"Generated {i+1}/{num_pairs} pairs")
    
    paired_z = torch.cat(paired_z, dim=0)
    paired_y = torch.cat(paired_y, dim=0)
    
    # 保存到文件，方便后续使用
    torch.save({'z': paired_z, 'y': paired_y}, 'paired_data.pth')
    print(f"Paired data saved to paired_data.pth")
    
    return paired_z, paired_y

# 在 train_dmd.py 开头添加
def visualize_paired_data(paired_z, paired_y, cfg):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(paired_y[:1000,0].cpu(), paired_y[:1000,1].cpu(), s=2, alpha=0.5)
    plt.title('Teacher outputs from paired data')
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    
    plt.subplot(1,2,2)
    plt.scatter(paired_z[:1000,0].cpu(), paired_z[:1000,1].cpu(), s=2, alpha=0.5)
    plt.title('Corresponding noise z')
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.savefig('paired_data_viz.png')
    print("Paired data visualization saved to paired_data_viz.png")


def train_dmd(cfg, teacher, diffusion):
    # 生成或加载 paired data
    paired_data_path = "paired_data.pth"
    if os.path.exists(paired_data_path) and not cfg.regenerate_paired:
        print(f"Loading paired data from {paired_data_path}")
        paired_data = torch.load(paired_data_path, map_location='cpu')
        paired_z = paired_data['z'].to(cfg.device)
        paired_y = paired_data['y'].to(cfg.device)
    else:
        print("Generating paired data...")
        paired_z, paired_y = generate_paired_data(teacher, diffusion, cfg, num_pairs=10000)
        paired_z = paired_z.to(cfg.device)
        paired_y = paired_y.to(cfg.device)
    
    visualize_paired_data(paired_z,paired_y,cfg)

    # 创建 dataset 和 dataloader
    paired_dataset = torch.utils.data.TensorDataset(paired_z, paired_y)
    paired_loader = torch.utils.data.DataLoader(
        paired_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        drop_last=True
    )
    paired_iter = iter(paired_loader)
    
    # ... 其余初始化代码 ...
    # 生成器 G
    G = Generator(
        z_dim=cfg.data_dim,          # 噪声维度与数据一致
        output_dim=cfg.data_dim,
        hidden_dim=cfg.hidden_dim*2,
        num_layers=cfg.num_layers
    ).to(cfg.device)

    # fake score 模型 μ_fake，初始化为 teacher 参数
    fake_model = MLP(
        input_dim=cfg.data_dim,
        output_dim=cfg.data_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        time_embed_dim=cfg.time_embed_dim
    ).to(cfg.device)
    fake_model.load_state_dict(teacher.state_dict())   # 从 teacher 拷贝

    # 优化器
    opt_G = optim.Adam(G.parameters(), lr=cfg.dmd_lr_G)
    opt_fake = optim.Adam(fake_model.parameters(), lr=cfg.dmd_lr_fake)

    # 损失函数（用于 fake 模型）
    mse = nn.MSELoss()

    # 固定 teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 训练循环
    for epoch in range(cfg.dmd_epochs):
        # --- 更新生成器 G ---
        G.train()
        fake_model.eval()

        # 采样噪声
        z = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
        x_fake = G(z)          # (batch, dim)

        # 随机采样时间步 t ∈ [t_min, t_max]
        t = torch.randint(int(cfg.t_min), int(cfg.t_max), (cfg.batch_size,), device=cfg.device)

        # 对 x_fake 加噪
        x_t, _ = diffusion.q_sample(x_fake, t)

        with torch.no_grad():
            # 教师预测
            pred_real = teacher(x_t, t)
            # fake 模型预测
            pred_fake = fake_model(x_t, t)

        # 计算 score
        alpha_t, sigma_t_sq = diffusion.get_alpha_sigma(t)   # (batch,1)
        s_real = -(x_t - alpha_t * pred_real) / sigma_t_sq
        s_fake = -(x_t - alpha_t * pred_fake) / sigma_t_sq

        # 梯度因子 (公式7)
        grad_factor = cfg.w_t * alpha_t * (s_fake - s_real).detach()   # (batch, dim)

        # 构造损失使得 dL/dx_fake = grad_factor
        loss_kl = (grad_factor * x_fake).sum() / cfg.batch_size
    
        # 在生成器更新部分
        try:
            z_ref, y_ref = next(paired_iter)
        except StopIteration:
            paired_iter = iter(paired_loader)
            z_ref, y_ref = next(paired_iter)

        z_ref = z_ref.to(cfg.device)
        y_ref = y_ref.to(cfg.device)

        x_ref = G(z_ref)
        loss_reg = F.mse_loss(x_ref, y_ref)  # 或使用 L1 loss

        # 计算 batch 内样本间的平均距离
        #z1 = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
        #z2 = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
        #x1 = G(z1)
        #x2 = G(z2)
        #pairwise_dist = torch.cdist(x1, x2).mean()

        # 添加到损失中（负号鼓励距离增大）
        #diversity_loss = -0.01 * pairwise_dist
        # 总损失
        total_loss_G = loss_kl + cfg.lambda_reg * loss_reg #+ diversity_loss 
        
        opt_G.zero_grad()
        total_loss_G.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        opt_G.step()

        # 计算一个可监控的代理指标（例如生成样本的方差）
        with torch.no_grad():
            z_test = torch.randn(100, cfg.data_dim, device=cfg.device)
            x_test = G(z_test)
            sample_var = x_test.var().item()

        #if epoch % 20 == 0:
        #    print(f"DMD Epoch {epoch}, Sample var: {sample_var:.4f}, Loss_fake: {loss_fake.item():.4f}")
        # --- 更新 fake 模型 (μ_fake) ---
        G.eval()
        fake_model.train()
        for Q in range(1,5):
            # 用当前生成器生成一批假样本（stop gradient）
            with torch.no_grad():
                z2 = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
                x_fake_detach = G(z2)

            # 随机采样时间步 t（全范围 0~T-1）
            t_fake = torch.randint(0, cfg.T, (cfg.batch_size,), device=cfg.device)
            x_t_fake, noise = diffusion.q_sample(x_fake_detach, t_fake)

            # 预测 x0
            pred = fake_model(x_t_fake, t_fake)

            # denoising loss (公式6)
            loss_fake = mse(pred, x_fake_detach)

            opt_fake.zero_grad()
            loss_fake.backward()
            torch.nn.utils.clip_grad_norm_(fake_model.parameters(), max_norm=1.0)
            opt_fake.step()

        if epoch % 20 == 0:
            print(f"DMD Epoch {epoch}, Sample var: {sample_var:.4f}, Loss: {total_loss_G.item():.4f}")

    return G, fake_model
