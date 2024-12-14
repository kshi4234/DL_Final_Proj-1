from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

class Encoder(nn.Module):
    """将输入观察编码为表征"""
    def __init__(self, input_channels=2, hidden_dim=256, output_dim=256):
        super().__init__()
        # CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 映射到表征空间
        self.fc = nn.Linear(hidden_dim * 4 * 4, output_dim)
        
    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        s = self.fc(h)
        return s  # [B, output_dim]

class Predictor(nn.Module):
    """基于当前表征和动作预测下一个表征"""
    def __init__(self, state_dim=256, action_dim=2, latent_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 状态动作编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # latent变量生成器
        self.latent_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # mean和std
        )
        
        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(512 + latent_dim, 512),
            nn.ReLU(), 
            nn.Linear(512, state_dim)
        )
        
    def forward(self, state, action, z=None):
        # state: [B, state_dim]
        # action: [B, action_dim]
        # z: [B, latent_dim] or None
        
        # 编码状态和动作
        h = self.encoder(torch.cat([state, action], dim=1))
        
        # 生成或使用latent
        if z is None:
            # 从分布中采样
            latent_params = self.latent_net(h)
            mean = latent_params[:, :self.latent_dim]
            std = F.softplus(latent_params[:, self.latent_dim:])
            z = mean + std * torch.randn_like(std)
        
        # 预测下一状态表征
        pred = self.predictor(torch.cat([h, z], dim=1))
        return pred, z

class JEPA(nn.Module):
    """完整的JEPA模型"""
    def __init__(self, 
                 repr_dim=256,    # 添加repr_dim作为属性
                 action_dim=2,
                 latent_dim=32,
                 use_target_encoder=True):
        super().__init__()
        
        # 保存表征维度作为属性
        self.repr_dim = repr_dim
        
        # 编码器
        self.encoder = Encoder(output_dim=repr_dim)
        
        # 目标编码器 (可以和主编码器共享参数)
        if use_target_encoder:
            self.target_encoder = Encoder(output_dim=repr_dim)
        else:
            self.target_encoder = self.encoder
            
        # 预测器
        self.predictor = Predictor(state_dim=repr_dim, action_dim=action_dim, latent_dim=latent_dim)
        
        # VICReg正则化器
        self.regularizer = VICRegRegularizer(repr_dim)

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Returns:
            predictions: [B, T, repr_dim]  # evaluator期望的格式
        """
        B = states.shape[0]
        
        # 如果是推理模式(evaluator调用)
        if states.shape[1] == 1:
            # 初始编码
            current_state = self.encoder(states[:, 0])  # [B, repr_dim]
            
            # 使用actions递归预测
            predictions = []
            for t in range(actions.shape[1]):
                pred_t, _ = self.predictor(current_state, actions[:, t])
                predictions.append(pred_t)
                current_state = pred_t
                
            predictions = torch.stack(predictions, dim=1)  # [B, T, repr_dim]
            return predictions

        # 训练模式 
        else:
            T = states.shape[1]
            
            # 初始编码
            init_state = self.encoder(states[:, 0])
            
            # 目标编码
            targets = []
            for t in range(1, T):
                target_t = self.target_encoder(states[:, t])
                targets.append(target_t)
            targets = torch.stack(targets, dim=1)
            
            # 递归预测
            predictions = []
            current_state = init_state
            for t in range(T-1):
                pred_t, _ = self.predictor(current_state, actions[:, t])
                predictions.append(pred_t)
                if self.training:
                    current_state = targets[:, t]  # teacher forcing
                else:
                    current_state = pred_t
            predictions = torch.stack(predictions, dim=1)
            
            # 计算正则化损失
            reg_loss = self.regularizer(predictions)
            
            return predictions, targets, reg_loss

class VICRegRegularizer(nn.Module):
    """VICReg正则化器防止表征坍缩"""
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim
        
    def forward(self, representations):
        # 方差损失
        std = torch.sqrt(representations.var(dim=0) + 1e-4)
        var_loss = F.relu(1 - std).mean()
        
        # 协方差损失
        cov = (representations - representations.mean(0)).T @ (representations - representations.mean(0))
        cov = cov / (representations.shape[0] - 1)
        cov_loss = (cov - torch.eye(self.repr_dim).to(cov.device)).pow(2).sum()
        
        return var_loss + 0.01 * cov_loss

def train_step(model, batch, optimizer):
    """单步训练"""
    states = batch.states  # [B, T, C, H, W]
    actions = batch.actions  # [B, T-1, 2]
    
    # 前向传播
    predictions, targets, reg_loss = model(states, actions)
    
    # 预测损失
    pred_loss = F.mse_loss(predictions, targets)
    
    # 总损失
    total_loss = pred_loss + reg_loss
    
    # 优化
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        'pred_loss': pred_loss.item(),
        'reg_loss': reg_loss.item(),
        'total_loss': total_loss.item()
    }