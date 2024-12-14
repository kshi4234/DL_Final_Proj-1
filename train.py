# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from dataset import create_wall_dataloader
from models import JEPA
import os
from tqdm import tqdm
import numpy as np

def train_jepa(
    batch_size=64,
    num_epochs=100,
    learning_rate=1e-4,
    device='cuda',
    save_path='checkpoints',
    repr_dim=256
):
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        batch_size=batch_size,
        train=True,
        device=device
    )
    
    # 初始化模型
    model = JEPA(repr_dim=repr_dim).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate * 0.01
    )
    
    # 记录最佳loss
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # 训练一个epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            # 前向传播
            predictions, targets, reg_loss = model(batch.states, batch.actions)
            
            # 计算预测损失
            pred_loss = F.mse_loss(predictions, targets)
            
            # 总损失
            loss = pred_loss + 0.1 * reg_loss  # 可调节正则化强度
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录loss
            train_losses.append(loss.item())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred_loss': f'{pred_loss.item():.4f}',
                'reg_loss': f'{reg_loss.item():.4f}'
            })
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均loss
        avg_loss = np.mean(train_losses)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(save_path, 'best_model.pth'))
            
        print(f'Epoch {epoch+1} Avg Loss: {avg_loss:.4f}')
        
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, os.path.join(save_path, 'final_model.pth'))
    
    print('Training completed!')
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_jepa(device=device)