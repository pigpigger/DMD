# config.py
import torch

class Config:
    # 数据
    data_dim = 2               # 点云维度
    num_components = 2         # 混合高斯成分数
    batch_size = 256
    num_samples = 10000         # 训练 teacher 的数据量

    # 扩散模型
    T = 400                     # 时间步数（简化，原论文 1000）
    beta_start = 1e-4
    beta_end = 0.02

    # 模型结构
    hidden_dim = 1024
    num_layers = 6
    time_embed_dim = 128

    # 训练 teacher
    teacher_epochs = 300
    teacher_lr = 2e-4

    # DMD 训练
    dmd_epochs = 600
    dmd_lr_G = 5e-5             # 生成器学习率
    dmd_lr_D = 5e-5             # 判别器学习率
    dmd_lr_fake = 5e-5           # fake score 模型学习率
    t_min = 0.1 * T             # 最小时间步
    t_max = 0.98 * T             # 最大时间步
    w_t = 0.05           # 简化的权重，可改为自适应

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    # 回归损失相关
    use_regression = True
    lambda_reg = 5
    regenerate_paired = False  # 设为 True 重新生成 paired data
    paired_data_size = 20000
    enable_f_div=False
    div_alpha=0
