import os, numpy as np
import torch
from torch.utils.data import Dataset

emotion_map = {
    'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5
}

class CREMADataset(Dataset):
    def __init__(self, processed_dir, augment=False):
        self.files = [f for f in os.listdir(processed_dir) if f.endswith('.npy')]
        self.processed_dir = processed_dir
        self.augment = augment
        
    def __len__(self): 
        return len(self.files)
    
    def _augment_data(self, x):
        """Apply data augmentation techniques"""
        if not self.augment:
            return x
            
        # Random noise addition
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise
            
        # Time shifting (circular shift)
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 10)
            x = np.roll(x, shift, axis=1)
            
        # Frequency masking (mask some frequency bins)
        if np.random.random() < 0.2:
            freq_mask_size = np.random.randint(1, 5)
            freq_start = np.random.randint(0, x.shape[0] - freq_mask_size)
            x[freq_start:freq_start + freq_mask_size, :] = 0
            
        # Time masking (mask some time frames)
        if np.random.random() < 0.2:
            time_mask_size = np.random.randint(1, 10)
            time_start = np.random.randint(0, x.shape[1] - time_mask_size)
            x[:, time_start:time_start + time_mask_size] = 0
            
        return x
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        emotion_code = fname.split('_')[2]
        x = np.load(os.path.join(self.processed_dir, fname))
        
        # Apply augmentation
        x = self._augment_data(x)
        
        # Normalize the features
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)  # Add channel dim
        y = emotion_map[emotion_code]
        return x, torch.tensor(y, dtype=torch.long)
