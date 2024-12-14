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
        
        # 仔细计算每一层的输出维度
        # 输入: [B, 2, 64, 64]
        self.conv = nn.Sequential(
            # -> [B, 32, 32, 32]
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # -> [B, 64, 16, 16]
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # -> [B, 128, 8, 8]
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # -> [B, 256, 4, 4]
            nn.Conv2d(128, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        
        # 4*4*256 = 4096
        self.fc = nn.Linear(hidden_dim * 4 * 4, output_dim)
        
    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)  # -> [B, hidden_dim, 4, 4]
        h = h.reshape(h.shape[0], -1)  # -> [B, hidden_dim * 4 * 4]
        s = self.fc(h)  # -> [B, output_dim]
        return s

def conv_output_size(input_size, kernel_size, stride, padding):
    """Helper function to calculate output size of conv layers"""
    return (input_size + 2*padding - kernel_size) // stride + 1



class JEPA(nn.Module):
    """完整的JEPA模型"""
    def __init__(self, 
                 repr_dim=256,    
                 action_dim=2,
                 latent_dim=32,
                 use_target_encoder=True):
        super().__init__()
        
        self.repr_dim = repr_dim
        
        # 编码器
        self.encoder = Encoder(output_dim=repr_dim)
        
        # 预测器
        self.predictor = Predictor(state_dim=repr_dim, action_dim=action_dim, latent_dim=latent_dim)
        
        # VICReg正则化器
        self.regularizer = VICRegRegularizer(repr_dim)

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
                actions: [B, T-1, 2]
            During inference (evaluator):
                states: [B, 1, Ch, H, W]
                actions: [B, T-1, 2]

        Returns:
            During training: 
                predictions, targets, reg_loss
            During inference:
                predictions: [B, T-1, repr_dim] 
        """
        B = states.shape[0]
        
        # 如果是evaluator调用(inference mode)
        if states.shape[1] == 1:
            # 初始编码
            current_state = self.encoder(states[:, 0])  # [B, repr_dim]
            
            # 预测T-1步
            predictions = []
            for t in range(actions.shape[1]):
                pred_t, _ = self.predictor(current_state, actions[:, t])
                predictions.append(pred_t)
                current_state = pred_t
            
            predictions = torch.stack(predictions, dim=1)  # [B, T-1, repr_dim]
            return predictions

        # 训练模式
        else:
            # ... 训练相关代码保持不变 ...
            pass

class Predictor(nn.Module):
    def __init__(self, state_dim=256, action_dim=2, latent_dim=32):
        super().__init__()
        
        # 状态动作编码器
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )
        
    def forward(self, state, action, z=None):
        # state: [B, state_dim]
        # action: [B, action_dim]
        x = torch.cat([state, action], dim=1)
        pred = self.net(x)
        return pred, None  # 为了简化，暂时不使用latent变量

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