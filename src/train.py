from dataset import CREMADataset
from model import SimpleCNN
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

ds = CREMADataset('../data/processed')
print(f"Total dataset size: {len(ds)} samples")

# Better train/validation split with stratification consideration
train_size = int(0.8 * len(ds))
train_ds, val_ds = random_split(ds, [train_size, len(ds)-train_size], 
                               generator=torch.Generator().manual_seed(42))

# Create augmented training dataset
train_indices = train_ds.indices
val_indices = val_ds.indices

# Create separate datasets with and without augmentation
train_ds_aug = CREMADataset('../data/processed', augment=True)
train_ds_aug.files = [ds.files[i] for i in train_indices]

val_ds_clean = CREMADataset('../data/processed', augment=False)
val_ds_clean.files = [ds.files[i] for i in val_indices]

train_loader = DataLoader(train_ds_aug, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds_clean, batch_size=32, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_ds_aug)}, Validation samples: {len(val_ds_clean)}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SimpleCNN()
model = model.to(device)

# Improved optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early stopping parameters
best_val_loss = float('inf')
patience = 7
patience_counter = 0
best_model_state = None

print("Starting training...")
print("=" * 60)

for epoch in range(50):  # Increased epochs with early stopping
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # Move to device
        
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)  # Move to device
            out = model(x)
            loss = loss_fn(out, y)
            
            val_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()
    
    # Calculate metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Print progress
    print(f'Epoch {epoch+1}/50:')
    print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Early stopping and best model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f'  *** New best validation loss: {best_val_loss:.4f} ***')
    else:
        patience_counter += 1
        print(f'  Patience: {patience_counter}/{patience}')
    
    print('-' * 60)
    
    # Early stopping
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs')
        break

print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")

# Save the best model
if best_model_state is not None:
    torch.save(best_model_state, '../models/emotion_model_best.pth')
    print("Best model saved to ../models/emotion_model_best.pth")

# Save the final model
torch.save(model.state_dict(), '../models/emotion_model_final.pth')
print("Final model saved to ../models/emotion_model_final.pth")
