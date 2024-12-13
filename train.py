import torch
import torch.nn.functional as F
from dataset import create_wall_dataloader
from models import JEPAModel

def vicreg_loss(x, y):
    # Invariance loss
    sim_loss = F.mse_loss(x, y)
    
    # Variance loss 
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
    
    # Covariance loss
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()
    
    return sim_loss + 25 * var_loss + 25 * cov_loss

def off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

def train(epochs=100):
    device = torch.device("cuda")
    model = JEPAModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    data_path = "/scratch/DL24FA/train"
    train_loader = create_wall_dataloader(
        data_path=data_path,
        device=device,
        batch_size=64,
        train=True
    )
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions # [B, T-1, 2]
            
            # Get predicted and target representations
            pred_states = model.encoder(states[:, :-1].reshape(-1, 2, 64, 64))
            with torch.no_grad():
                target_states = model.target_encoder(states[:, 1:].reshape(-1, 2, 64, 64))
            
            # Predict next states
            pred_next = model.predictor(pred_states, actions.reshape(-1, 2))
            
            # Compute loss with collapse prevention
            loss = vicreg_loss(pred_next, target_states.detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target encoder
            model.update_target()
            
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    train()
