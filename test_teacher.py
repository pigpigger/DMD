# test_teacher.py
import torch
import matplotlib.pyplot as plt
import os
from config import Config
from train_teacher import train_teacher
from data import generate_mixture_of_gaussians

def visualize_results(teacher, diffusion, cfg):
    teacher.eval()
    
    #real_data = generate_mixture_of_gaussians(1000, cfg.data_dim, cfg.num_components, cfg.device)
     
    with torch.no_grad():
        # 从标准高斯噪声开始
        x = torch.randn(10000, cfg.data_dim, device=cfg.device)
        alpha_bars = diffusion.alpha_bars

        # 确定性 DDIM 采样循环
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
            pred_noise = teacher(x, t_tensor)

            # 再由噪声恢复 x0
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / (sqrt_alpha_bar_t + 1e-8)


            #dir_xt = (x - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t

            # 3. 计算 x_{t-1} (确定性更新, sigma_t = 0)
            # 公式: x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + sqrt(1 - alpha_bar_{t-1}) * dir_xt
            #x = sqrt_alpha_bar_t_prev * pred_x0 + sqrt_one_minus_alpha_bar_t_prev * dir_xt
            x = sqrt_alpha_bar_t_prev * pred_x0 + sqrt_one_minus_alpha_bar_t_prev * pred_noise
            # 可选：数值稳定性检查
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"NaN/Inf detected at t={t}")
                print(f"pred_x0 range: [{pred_x0.min().item():.4f}, {pred_x0.max().item():.4f}]")
                print(f"dir_xt range: [{dir_xt.min().item():.4f}, {dir_xt.max().item():.4f}]")
                print(f"sqrt_one_minus_alpha_bar_t: {sqrt_one_minus_alpha_bar_t.item()}")
                # 遇到 NaN 直接中断，避免后续计算无意义
                break

        generated = x
    # 打印最终统计
    print(f"Final generated: mean={generated.mean().item():.4f}, std={generated.std().item():.4f}")
    #print(f"Real data: mean={real_data.mean().item():.4f}, std={real_data.std().item():.4f}")
    
    plt.figure(figsize=(8, 6))
    
    if not torch.isnan(generated).any():
        plt.scatter(generated[:,0].cpu(), generated[:,1].cpu(), 
                   s=2, alpha=0.5, c='red')
        plt.title('Teacher Generated Samples')
    else:
        plt.text(0.5, 0.5, 'All NaN', ha='center', va='center', fontsize=20)
        plt.title('Generation Failed')
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('teacher_test.png', dpi=100)
    plt.close()
    print("生成结果已保存到 teacher_test.png")

def main():
    # 创建配置
    cfg = Config()
    print(f"使用设备: {cfg.device}")
    print(f"数据维度: {cfg.data_dim}")
    
    # 检查是否有保存的模型
    model_path = "teacher.pth"
    retrain = False  # 是否重新训练
    
    if os.path.exists(model_path) and not retrain:
        print(f"找到已保存的模型 {model_path}，直接加载...")
        
        # 创建模型并加载参数
        from models import MLP
        from diffusion import Diffusion
        
        teacher = MLP(
            input_dim=cfg.data_dim,
            output_dim=cfg.data_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            time_embed_dim=cfg.time_embed_dim
        ).to(cfg.device)
        
        teacher.load_state_dict(torch.load(model_path, map_location=cfg.device))
        teacher.eval()
        
        diffusion = Diffusion(cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)
        
        print("模型加载成功！")
        
    else:
        if not os.path.exists(model_path):
            print(f"未找到模型文件 {model_path}，开始训练...")
        else:
            print("重新训练模型...")
        
        teacher, diffusion = train_teacher(cfg)
        torch.save(teacher.state_dict(), model_path)
        print(f"teacher 模型已保存到 {model_path}")
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(teacher, diffusion, cfg)
    
    print("\n完成！")

if __name__ == "__main__":
    main()
