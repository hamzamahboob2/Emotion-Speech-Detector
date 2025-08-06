import torch
import numpy as np
import librosa
from model import SimpleCNN
from dataset import emotion_map
import os

class EmotionPredictor:
    def __init__(self, model_path='../models/emotion_model_best.pth'):
        """Initialize the emotion predictor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SimpleCNN()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from: {model_path}")
        except FileNotFoundError:
            print(f"Model {model_path} not found. Please train the model first.")
            return
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Emotion mapping
        self.emotion_names = list(emotion_map.keys())
        
    def extract_features(self, audio_path, n_mfcc=40, max_length=100):
        """Extract MFCC features from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Pad or truncate to fixed length
            if mfcc.shape[1] < max_length:
                pad_width = max_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_length]
            
            # Normalize
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            
            return mfcc
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def predict_emotion(self, audio_path, return_probabilities=False):
        """Predict emotion from audio file"""
        # Extract features
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predicted_emotion = self.emotion_names[predicted.item()]
        confidence = probabilities[0][predicted].item()
        
        if return_probabilities:
            prob_dict = {emotion: prob.item() for emotion, prob in zip(self.emotion_names, probabilities[0])}
            return predicted_emotion, confidence, prob_dict
        
        return predicted_emotion, confidence
    
    def predict_batch(self, audio_files):
        """Predict emotions for multiple audio files"""
        results = []
        for audio_file in audio_files:
            result = self.predict_emotion(audio_file, return_probabilities=True)
            if result:
                results.append({
                    'file': os.path.basename(audio_file),
                    'predicted_emotion': result[0],
                    'confidence': result[1],
                    'probabilities': result[2]
                })
            else:
                results.append({
                    'file': os.path.basename(audio_file),
                    'error': 'Failed to process'
                })
        return results

def demo_predictions():
    """Demo function to test predictions on sample files"""
    predictor = EmotionPredictor()
    
    # Get some sample audio files
    audio_dir = '../data/raw/AudioWAV'
    sample_files = []
    
    if os.path.exists(audio_dir):
        all_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        # Sample 10 files from different emotions
        sample_files = all_files[:10]  # Take first 10 files
        
        print("Sample Predictions:")
        print("=" * 60)
        
        for audio_file in sample_files:
            file_path = os.path.join(audio_dir, audio_file)
            result = predictor.predict_emotion(file_path, return_probabilities=True)
            
            if result:
                pred_emotion, confidence, prob_dict = result
                true_emotion = audio_file.split('_')[2]  # Extract true emotion from filename
                
                print(f"File: {audio_file}")
                print(f"True Emotion: {true_emotion}")
                print(f"Predicted: {pred_emotion} (confidence: {confidence:.3f})")
                print(f"All probabilities: {', '.join([f'{e}:{p:.3f}' for e, p in prob_dict.items()])}")
                print(f"Correct: {'✓' if pred_emotion == true_emotion else '✗'}")
                print("-" * 60)
    else:
        print("Audio directory not found. Please check the path.")

if __name__ == "__main__":
    demo_predictions()
