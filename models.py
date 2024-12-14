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
    def __init__(self, input_channels=2, hidden_dim=256, output_dim=256):
        super().__init__()
        
        # 更深的CNN backbone
        self.conv = nn.Sequential(
            # 1st block
            nn.Conv2d(input_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 2nd block
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 3rd block
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        s = self.fc(h)
        return s


class JEPA(nn.Module):
    def __init__(self, repr_dim=256, input_channels=2):
        super().__init__()
        self.repr_dim = repr_dim
        
        # Initialize components
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_dim=256,
            output_dim=repr_dim
        )
        
        self.predictor = Predictor(
            state_dim=repr_dim,
            action_dim=2
        )
        
        self.regularizer = VICRegRegularizer(repr_dim)

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
                actions: [B, T-1, 2]
            During inference:
                states: [B, 1, Ch, H, W]
                actions: [B, T-1, 2]
                
        Returns: 
            训练模式: predictions [B, T, repr_dim], targets [B, T, repr_dim], reg_loss
            推理模式: predictions [B, T, repr_dim]  # 注意这里改为T而不是T-1
        """
        B = states.shape[0]
        
        # 如果是evaluator调用(inference mode)
        if states.shape[1] == 1:
            # 初始编码
            current_state = self.encoder(states[:, 0])  # [B, repr_dim]
           
            # 预测T步
            predictions = [current_state]  # 从初始状态开始
            for t in range(actions.shape[1]):
                pred_t, _ = self.predictor(current_state, actions[:, t])
                predictions.append(pred_t)
                current_state = pred_t
           
            predictions = torch.stack(predictions, dim=1)  # [B, T, repr_dim]
            return predictions
        
        # 训练模式代码保持不变
        else:
            T = states.shape[1]
            
            # 初始编码
            init_state = self.encoder(states[:, 0])  # [B, repr_dim]
            
            # 编码目标状态
            targets = []
            for t in range(1, T):
                target_t = self.encoder(states[:, t])  # [B, repr_dim]
                targets.append(target_t)
            targets = torch.stack(targets, dim=1)  # [B, T-1, repr_dim]
            
            # 预测未来状态
            predictions = []
            current_state = init_state
            for t in range(T-1):
                pred_t, _ = self.predictor(current_state, actions[:, t])
                predictions.append(pred_t)
                if self.training:
                    current_state = targets[:, t]  # teacher forcing
                else:
                    current_state = pred_t
            predictions = torch.stack(predictions, dim=1)  # [B, T-1, repr_dim]
            
            reg_loss = self.regularizer(predictions)
            
            return predictions, targets, reg_loss

class Predictor(nn.Module):
    def __init__(self, state_dim=256, action_dim=2, latent_dim=32):
        super().__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # latent变量生成器
        self.latent_net = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # mean和std
        )
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(512 + 64 + latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )
        
    def forward(self, state, action, z=None):
        # 编码state和action
        state_enc = self.state_encoder(state)
        action_enc = self.action_encoder(action)
        combined = torch.cat([state_enc, action_enc], dim=1)
        
        # 生成或使用latent
        if z is None:
            latent_params = self.latent_net(combined)
            mean = latent_params[:, :self.latent_dim]
            std = F.softplus(latent_params[:, self.latent_dim:])
            z = mean + std * torch.randn_like(std) if self.training else mean
            
        # 预测下一状态
        pred = self.predictor(torch.cat([combined, z], dim=1))
        return pred, z

class VICRegRegularizer(nn.Module):
    """VICReg正则化器防止表征坍缩"""
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim
        
    def forward(self, representations):
        """
        Args:
            representations: [B, T, D] or [B, D]
        """
        # 如果是3D张量，展平成2D
        if representations.dim() == 3:
            B, T, D = representations.shape
            representations = representations.reshape(-1, D)  # [B*T, D]
            
        # 计算每个维度的标准差
        std = torch.sqrt(representations.var(dim=0) + 1e-4)
        var_loss = F.relu(1 - std).mean()
        
        # 中心化
        representations = representations - representations.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵 [D, D]
        N = representations.shape[0]  # batch size or batch_size * timesteps
        cov = (representations.T @ representations) / (N - 1)
        
        # 与单位矩阵的差异
        cov_loss = off_diagonal(cov).pow(2).sum()
        
        return var_loss + 0.01 * cov_loss

def off_diagonal(x):
    """返回一个矩阵的非对角线元素"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()