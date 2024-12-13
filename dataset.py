from typing import NamedTuple, Optional
import torch
import numpy as np


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        self.device = device
        # Use memmap to avoid loading entire array into memory
        self.states = np.load(f"{data_path}/states.npy", mmap_mode='r')
        self.actions = np.load(f"{data_path}/actions.npy", mmap_mode='r')
        if probing:
            self.locations = np.load(f"{data_path}/locations.npy", mmap_mode='r')
        else:
            self.locations = None
        self.probing = probing

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        # Load and convert small chunks as needed
        states = torch.from_numpy(self.states[i].copy()).float().to(self.device)
        actions = torch.from_numpy(self.actions[i].copy()).float().to(self.device)
        
        if self.probing and self.locations is not None:
            locations = torch.from_numpy(self.locations[i].copy()).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)
        
        return WallSample(states=states, locations=locations, actions=actions)

def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader
