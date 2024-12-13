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


def vicreg_loss(x, y, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
    """Improved VICReg loss with better coefficients and normalized distance"""
    # Invariance loss (normalized)
    sim_loss = F.smooth_l1_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    
    # Variance loss (encourage spread)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
    
    # Covariance loss (decorrelation)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()
    
    return sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            # Input: [B, 2, 64, 64]
            nn.Conv2d(2, 64, 4, stride=2, padding=1),    # More channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim)  # Normalize output
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim)  # Normalize output
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
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
        curr_state = self.encoder(states.squeeze(1))  # [B, D]
        predictions = [curr_state]
        
        # Predict future states
        for t in range(T-1):
            curr_state = self.predictor(curr_state, actions[:, t])
            predictions.append(curr_state)
            
        return torch.stack(predictions, dim=1)  # [B, T, D]
