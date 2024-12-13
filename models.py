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
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            # Input: [B, 2, 64, 64]
            nn.Conv2d(2, 32, 3, stride=2, padding=1),    # [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # [B, 256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Calculate correct input size for fc layer: 256 channels * 4 * 4
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # Print shape for debugging
        B = x.shape[0]
        x = self.conv(x)  # [B, 256, 4, 4]
        x = x.reshape(B, -1)  # [B, 256*4*4]
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.LayerNorm(512), 
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
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
