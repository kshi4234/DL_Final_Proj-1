from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math  # Add this import


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


def vicreg_loss(x, y, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
    # Invariance loss (normalized)
    sim_loss = F.smooth_l1_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    
    # Variance loss with stronger regularization
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(2.0 - std_x)) + torch.mean(F.relu(2.0 - std_y))
    
    # Covariance loss with normalized features
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()
    
    return sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn = self.conv(x)
        return x * attn


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        
        # Initialize position encoding after model is created
        self.register_buffer('pos_encoding', None)
        
        self.layer1 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            SpatialAttention(128)
        )
        
        self.layer2 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            SpatialAttention(256)
        )
        
        self.layer3 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            SpatialAttention(512)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add position MLP head
        self.pos_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2)  # Predict normalized x,y coordinates
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 + 2, latent_dim),  # Concat position features
            nn.LayerNorm(latent_dim)
        )
        
    def _create_pos_encoding(self, x):
        """Create position encoding matching input feature map size"""
        _, _, h, w = x.shape  # Get actual dimensions after first conv
        device = x.device
        
        # Create normalized coordinate grid
        pos_h = torch.arange(h, device=device).float().view(-1, 1).repeat(1, w) / h
        pos_w = torch.arange(w, device=device).float().view(1, -1).repeat(h, 1) / w
        pos = torch.stack([pos_h, pos_w], dim=0)  # [2, H, W]
        pos = pos.unsqueeze(0)  # [1, 2, H, W]
        
        # Project to match channel dimension using fixed random projection
        projection = torch.randn(64, 2, 1, 1, device=device) / math.sqrt(2)
        pos = F.conv2d(pos, projection, padding=0)  # [1, 64, H, W]
        
        return pos
        
    def forward(self, x):
        x = self.conv1(x)
        
        # Create or update position encoding if needed
        if self.pos_encoding is None or self.pos_encoding.shape[-2:] != x.shape[-2:]:
            self.pos_encoding = self._create_pos_encoding(x)
            
        # Add position encoding
        x = x + self.pos_encoding.to(x.device)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        feat = self.avg_pool(x)
        feat = feat.view(feat.size(0), -1)
        
        # Get position prediction
        pos_pred = self.pos_head(feat)  # Store position prediction
        
        # Combine features with position
        x = torch.cat([feat, pos_pred], dim=1)
        x = self.fc(x)
        return x  # Only return latent representation for compatibility


class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Add collision prediction
        self.collision_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        feat = self.net(x)
        
        # Predict collision probability
        collision = self.collision_head(feat)
        
        # If collision predicted, reduce action magnitude
        action_scale = 1.0 - 0.9 * collision
        scaled_action = action * action_scale
        
        # Recompute with scaled action
        x = torch.cat([state, scaled_action], dim=-1)
        return self.net(x)


class JEPAModel(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.target_encoder = Encoder(latent_dim)
        self.repr_dim = latent_dim
        
        # Initialize target encoder
        for param_q, param_k in zip(self.encoder.parameters(), 
                                  self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for param_q, param_k in zip(self.encoder.parameters(),
                                  self.target_encoder.parameters()):
            param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data
            
    def forward(self, states, actions):
        """
        During inference:
            states: [B, 1, Ch, H, W] - initial state only
            actions: [B, T-1, 2] - sequence of actions
        Returns:
            predictions: [B, T, D] - predicted representations
        """
        B = states.shape[0]
        T = actions.shape[1] + 1
        D = self.repr_dim
        
        # Get initial embedding
        curr_state, _ = self.encoder(states.squeeze(1))  # [B, D]
        predictions = [curr_state]
        
        # Predict future states
        for t in range(T-1):
            curr_state = self.predictor(curr_state, actions[:, t])
            predictions.append(curr_state)
            
        return torch.stack(predictions, dim=1)  # [B, T, D]
