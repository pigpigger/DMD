import torch
import torch.nn as nn
import torch.optim as optim
from diffusion import Diffusion
from models import MLP
from data import get_dataloader

def train_teacher(cfg):
    loader = get_dataloader(
        cfg.batch_size, cfg.num_samples, cfg.data_dim, cfg.num_components, cfg.device
    )

    diffusion = Diffusion(cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)

    teacher = MLP(
        input_dim=cfg.data_dim,
        output_dim=cfg.data_dim,   # 仍然输出 data_dim，这里表示预测 epsilon
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        time_embed_dim=cfg.time_embed_dim
    ).to(cfg.device)

    optimizer = optim.Adam(teacher.parameters(), lr=cfg.teacher_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.teacher_epochs
    )
    mse = nn.MSELoss()

    teacher.train()
    for epoch in range(cfg.teacher_epochs):
        total_loss = 0.0

        for batch in loader:
            x0 = batch[0].to(cfg.device)
            batch_sz = x0.shape[0]

            #t = torch.randint(0, cfg.T, (batch_sz,), device=cfg.device)
            t = torch.randint(int(0.02*cfg.T), int(0.8*cfg.T),(batch_sz,), device=cfg.device)
            noise = torch.randn_like(x0)
            x_t, _ = diffusion.q_sample(x0, t, noise)

            # 关键改动：teacher 预测的是 noise / epsilon
            pred_noise = teacher(x_t, t)

            loss = mse(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

    return teacher, diffusion
