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

def train(epochs=2):  # Increase epochs
    device = torch.device("cuda")
    model = JEPAModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Slightly higher learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    
    data_path = "/scratch/DL24FA/train"
    train_loader = create_wall_dataloader(
        data_path=data_path,
        device=device,
        batch_size=128,  # Larger batch size
        train=True
    )
    
    best_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            states = batch.states
            actions = batch.actions
            
            B, T, C, H, W = states.shape
            
            # Convert states to proper shape for encoder
            curr_states = states[:, :-1].contiguous()
            next_states = states[:, 1:].contiguous()
            
            # Flatten batch and time dimensions
            curr_states = curr_states.view(-1, C, H, W)
            next_states = next_states.view(-1, C, H, W)
            
            # Get embeddings
            pred_states = model.encoder(curr_states)
            with torch.no_grad():
                target_states = model.target_encoder(next_states)
            
            actions_flat = actions.reshape(-1, 2)
            
            # Predict next states
            pred_next = model.predictor(pred_states, actions_flat)
            
            # Compute loss
            loss = vicreg_loss(pred_next, target_states.detach())
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update target encoder
            model.update_target(momentum=0.996)  # Slightly higher momentum
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if num_batches % 200 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    train()
