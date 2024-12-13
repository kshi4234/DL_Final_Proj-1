import torch
import torch.nn.functional as F
from dataset import create_wall_dataloader
from models import JEPAModel
from tqdm import tqdm
import random
import math
from normalizer import Normalizer  # Add this import

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

def train(epochs=50):  # Increase epochs significantly
    device = torch.device("cuda")
    model = JEPAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Remove scheduler and use custom learning rate adjustment
    warmup_steps = 1000
    total_steps = 0  # Will be set after creating dataloader
    
    data_path = "/scratch/DL24FA/train"
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=True,  # Changed to True to get locations
        device=device,
        batch_size=128,  # Larger batch size
        train=True
    )
    
    total_steps = epochs * len(train_loader)
    best_loss = float('inf')
    step = 0
    
    # Add these variables for tracking
    total_batches = len(train_loader)
    best_epoch_loss = float('inf')
    
    model.train()
    # Add progress bar for epochs
    pbar_epoch = tqdm(range(epochs), desc='Training epochs')
    
    for epoch in pbar_epoch:
        epoch_loss = 0
        num_batches = 0
        
        # Add progress bar for batches within each epoch
        pbar_batch = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch_idx, batch in enumerate(pbar_batch):
            # Get current learning rate
            if step < warmup_steps:
                curr_lr = 2e-4 * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                curr_lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * progress))
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
                
            # Rest of training loop
            states = batch.states
            actions = batch.actions
            
            # Add more augmentations
            if random.random() < 0.5:
                states = torch.flip(states, [3])  # Horizontal flip
                actions[:,:,0] = -actions[:,:,0]
                
            if random.random() < 0.3:
                # Random crop and resize
                b, t, c, h, w = states.shape
                states = states.view(-1, c, h, w)
                crop_size = random.randint(48, 64)
                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                states = F.interpolate(
                    states[:, :, i:i+crop_size, j:j+crop_size],
                    size=(64, 64),
                    mode='bilinear'
                )
                states = states.view(b, t, c, 64, 64)
                
            # Add Gaussian noise
            if random.random() < 0.3:
                states = states + torch.randn_like(states) * 0.01
                
            B, T, C, H, W = states.shape
            curr_states = states[:, :-1].contiguous().view(-1, C, H, W)
            next_states = states[:, 1:].contiguous().view(-1, C, H, W)
            
            pred_states = model.encoder(curr_states)
            with torch.no_grad():
                target_states = model.target_encoder(next_states)
            
            actions_flat = actions.reshape(-1, 2)
            
            # Get wall channel from states for collision detection
            wall_channel = states[:, :, 1:2]  # Get wall channel
            collision_mask = F.max_pool2d(wall_channel.view(-1, 1, H, W), 3, stride=1, padding=1) > 0.5
            
            # Calculate auxiliary position loss
            if hasattr(batch, 'locations') and batch.locations.numel() > 0:
                normalizer = Normalizer()
                true_positions = batch.locations
                normalized_positions = normalizer.normalize_location(true_positions)
                
                # Get position predictions from encoder
                all_states = states.view(-1, C, H, W)
                encoded = model.encoder(all_states)
                pos_loss = F.mse_loss(encoded[:, -2:], normalized_positions.view(-1, 2))
            else:
                pos_loss = 0.0
            
            # Collision-aware prediction loss
            pred_next = model.predictor(pred_states, actions_flat)
            pred_collision = model.predictor.collision_head(pred_next)
            collision_loss = F.binary_cross_entropy(pred_collision, collision_mask.float())
            
            # Combined loss
            loss = vicreg_loss(pred_next, target_states.detach())
            if pos_loss != 0.0:
                loss = loss + 0.1 * pos_loss
            loss = loss + 0.1 * collision_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update momentum based on training progress
            momentum = min(0.996, 0.99 + step/total_steps * 0.006)
            model.update_target(momentum=momentum)
            
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            # Print detailed training stats every 300 batches
            if batch_idx % 300 == 0:
                tqdm.write(
                    f"[Epoch {epoch}/{epochs}][Batch {batch_idx}/{total_batches}] "
                    f"Loss: {loss.item():.4f}, LR: {curr_lr:.6f}, "
                    f"Position Loss: {pos_loss if isinstance(pos_loss, float) else pos_loss.item():.4f}, "
                    f"Collision Loss: {collision_loss.item():.4f}"
                )
            
            pbar_batch.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{curr_lr:.6f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        pbar_epoch.set_postfix({'avg_loss': f'{avg_epoch_loss:.4f}'})
        
        # Save best model and print epoch summary
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "model_weights.pth")  # Changed from best_model.pth
            tqdm.write(
                f"\nEpoch {epoch} Summary:\n"
                f"Average Loss: {avg_epoch_loss:.4f} (New Best)\n"
                f"Learning Rate: {curr_lr:.6f}\n"
            )
        else:
            tqdm.write(
                f"\nEpoch {epoch} Summary:\n"
                f"Average Loss: {avg_epoch_loss:.4f}\n"
                f"Learning Rate: {curr_lr:.6f}\n"
            )
    
    print("\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    train()
