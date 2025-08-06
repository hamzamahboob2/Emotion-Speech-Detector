import os, librosa, numpy as np

def extract_features(file_path, n_mfcc=40, max_length=100):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_length:
        # Pad with zeros
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate
        mfcc = mfcc[:, :max_length]
    
    return mfcc

def process_crema_dataset(raw_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(raw_dir):
        if not fname.endswith('.wav'): continue
        try:
            feat = extract_features(os.path.join(raw_dir, fname))
            np.save(os.path.join(dest_dir, fname.replace('.wav', '.npy')), feat)
        except Exception as e:
            print("Error processing", fname, ":", e)

if __name__ == "__main__":
    # Define paths relative to the project root
    raw_audio_dir = "../data/raw/AudioWAV"
    processed_dir = "../data/processed"
    
    print("Starting audio preprocessing...")
    print(f"Raw audio directory: {raw_audio_dir}")
    print(f"Processed data directory: {processed_dir}")
    
    # Process the dataset
    process_crema_dataset(raw_audio_dir, processed_dir)
    print("Audio preprocessing completed!")
