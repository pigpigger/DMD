# main.py
from config import Config
from train_teacher import train_teacher
from train_dmd import train_dmd
from utils import visualize_2d
import torch
import os
def main():
    cfg = Config()
    
    # 选项设置
    USE_EXISTING_TEACHER = True  # 设为 True 表示使用已有 teacher.pth，False 表示重新训练
    
    # 检查是否存在 teacher 模型
    teacher_path = "teacher.pth"
    teacher_exists = os.path.exists(teacher_path)
    
    if USE_EXISTING_TEACHER and teacher_exists:
        print(f"找到已保存的 teacher 模型 {teacher_path}，直接加载...")
        
        # 创建模型结构
        from models import MLP
        teacher = MLP(
            input_dim=cfg.data_dim,
            output_dim=cfg.data_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            time_embed_dim=cfg.time_embed_dim
        ).to(cfg.device)
        
        # 加载参数
        teacher.load_state_dict(torch.load(teacher_path, map_location=cfg.device, weights_only=True))
        teacher.eval()
        
        # 创建 diffusion 对象（不需要重新训练）
        from diffusion import Diffusion
        diffusion = Diffusion(cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)
        
        print("Teacher 模型加载成功！")
        
    else:
        if not teacher_exists:
            print(f"未找到 teacher 模型 {teacher_path}，开始训练...")
        else:
            print("USE_EXISTING_TEACHER 为 False，重新训练 teacher...")
        
        # 训练 teacher 扩散模型
        print("Training teacher diffusion...")
        teacher, diffusion = train_teacher(cfg)
        
        # 保存 teacher
        torch.save(teacher.state_dict(), teacher_path)
        print(f"Teacher saved to {teacher_path}")

    # 2. DMD 训练
    print("\nTraining DMD...")
    G, fake_model = train_dmd(cfg, teacher, diffusion)

    # 保存生成器
    #torch.save(G.state_dict(), "generator.pth")
    #print("Generator saved to generator.pth")

    # 3. 生成一些样本并可视化（仅当 dim=2 时）
    if cfg.data_dim == 2:
        G.eval()
        with torch.no_grad():
            z = torch.randn(10000, cfg.data_dim, device=cfg.device)
            samples = G(z).cpu()
        visualize_2d(samples, "dmd_samples.png")
        print("Generated samples saved to dmd_samples.png")

if __name__ == "__main__":
    main()

