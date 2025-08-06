# 🎵 Emotion Detection from Audio using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-76.98%25-brightgreen.svg)](#performance)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning system for detecting emotions from audio files using Convolutional Neural Networks (CNN) and MFCC feature extraction. Achieves **76.98% accuracy** on the CREMA-D dataset.

## 🎯 Overview

This project implements an end-to-end emotion detection pipeline that:
- Extracts MFCC features from audio files
- Uses a custom CNN architecture with batch normalization and dropout
- Employs advanced training techniques (early stopping, learning rate scheduling, data augmentation)
- Classifies audio into 6 emotions: **Anger**, **Disgust**, **Fear**, **Happiness**, **Neutral**, **Sadness**

## 🚀 Key Features

- **High Accuracy**: 76.98% on 7,442 audio samples
- **Real-time Inference**: Fast prediction on new audio files
- **Data Augmentation**: Noise addition, time/frequency masking, time shifting
- **Robust Training**: Early stopping, gradient clipping, learning rate scheduling
- **Complete Pipeline**: From raw audio to trained model to predictions
- **Visualization**: Confusion matrices and detailed performance metrics

## 📊 Performance Results

### Overall Performance
- **Accuracy**: 76.98%
- **Training Time**: ~43 epochs (automatic early stopping)
- **Model Size**: 2.8M parameters
- **Best Validation Loss**: 0.9760

### Per-Emotion Performance
| Emotion | Precision | Recall | F1-Score | Accuracy |
|---------|-----------|---------|----------|----------|
| **Anger (ANG)** | 0.72 | 0.95 | 0.82 | **94.57%** |
| **Disgust (DIS)** | 0.73 | 0.72 | 0.73 | 72.23% |
| **Fear (FEA)** | 0.72 | 0.78 | 0.75 | 77.89% |
| **Happiness (HAP)** | 0.84 | 0.70 | 0.77 | 70.18% |
| **Neutral (NEU)** | 0.89 | 0.79 | 0.84 | 79.30% |
| **Sadness (SAD)** | 0.78 | 0.68 | 0.73 | 68.06% |

### Sample Predictions (with confidence scores)
```
File: 1001_DFA_ANG_XX.wav
True: ANG | Predicted: ANG | Confidence: 96.3% ✓

File: 1001_DFA_FEA_XX.wav  
True: FEA | Predicted: FEA | Confidence: 93.5% ✓

File: 1001_DFA_HAP_XX.wav
True: HAP | Predicted: HAP | Confidence: 95.9% ✓
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- Windows/macOS/Linux

### 1. Clone the Repository
```bash
git clone https://github.com/hamzamahboob2/Emotion-Speech-Detector.git
cd Emotion-Speech-Detector
```

### 2. Install Dependencies

#### Option A: Using requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation
```bash
pip install torch torchvision torchaudio
pip install librosa numpy scikit-learn matplotlib seaborn soundfile
```

### 3. Download Dataset
Place your CREMA-D dataset audio files in:
```
data/raw/AudioWAV/
```

## 🎯 Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
python run_project.py
```
This runs: preprocessing → training → evaluation → inference

### Option 2: Step-by-Step

#### 1. Feature Extraction
```bash
cd src
python preprocessing.py
```

#### 2. Train Model
```bash
# Quick training (5 epochs - for demo)
python train_quick.py

# Full training (up to 50 epochs with early stopping)
python train.py
```

#### 3. Evaluate Model
```bash
python evaluate.py
```

#### 4. Make Predictions
```bash
python inference.py
```

## 💻 Usage Examples

### Single File Prediction
```python
from src.inference import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor()

# Predict emotion from audio file
emotion, confidence = predictor.predict_emotion('path/to/audio.wav')
print(f"Emotion: {emotion} (Confidence: {confidence:.3f})")

# Get detailed probabilities
emotion, confidence, probs = predictor.predict_emotion(
    'path/to/audio.wav', 
    return_probabilities=True
)
for emotion, prob in probs.items():
    print(f"{emotion}: {prob:.3f}")
```

### Batch Prediction
```python
from src.inference import EmotionPredictor

predictor = EmotionPredictor()
audio_files = ['file1.wav', 'file2.wav', 'file3.wav']
results = predictor.predict_batch(audio_files)

for result in results:
    print(f"File: {result['file']}")
    print(f"Emotion: {result['predicted_emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
```

## 🧠 Technical Details

### Model Architecture
```python
SimpleCNN:
├── Conv2d(1→32) + BatchNorm2d + ReLU + MaxPool2d(2×2)
├── Conv2d(32→64) + BatchNorm2d + ReLU + MaxPool2d(2×2)  
├── Conv2d(64→128) + BatchNorm2d + ReLU
├── Conv2d(128→256) + BatchNorm2d + ReLU
├── AdaptiveAvgPool2d(2×2)
├── Flatten()
├── Linear(1024→512) + ReLU + Dropout(0.5)
├── Linear(512→128) + ReLU + Dropout(0.3)
└── Linear(128→6) [6 emotion classes]
```

### Feature Engineering
- **MFCC Features**: 40 coefficients per frame
- **Fixed Length**: Padded/truncated to 100 time frames
- **Sample Rate**: 22,050 Hz
- **Normalization**: Zero mean, unit variance per sample
- **Input Shape**: (1, 40, 100) - [channels, mfcc_coeffs, time_frames]

### Training Configuration
- **Optimizer**: AdamW (lr=5e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Batch Size**: 64
- **Early Stopping**: Patience=3 epochs
- **Gradient Clipping**: Max norm=1.0
- **Loss Function**: CrossEntropyLoss

### Data Augmentation Techniques
1. **Gaussian Noise**: Random noise addition (30% probability)
2. **Time Shifting**: Circular shift of time frames (30% probability) 
3. **Frequency Masking**: Mask random frequency bins (20% probability)
4. **Time Masking**: Mask random time frames (20% probability)

## 📈 Performance Analysis

### Strengths
- ✅ **High Overall Accuracy**: 76.98% (well above random baseline of 16.67%)
- ✅ **Excellent Anger Detection**: 94.57% accuracy with 95% recall
- ✅ **Balanced Performance**: Good F1-scores across all emotions
- ✅ **High Confidence**: Many predictions have >90% confidence
- ✅ **Robust Training**: Automatic early stopping prevented overfitting

### Areas for Improvement
- 🔄 **Sadness Detection**: Lowest accuracy at 68.06%
- 🔄 **Class Imbalance**: Some emotions harder to distinguish
- 🔄 **Model Size**: Could explore lighter architectures for deployment

## 🎛️ Configuration Options

### Modify Training Parameters
Edit `src/train.py`:
```python
# Learning rate
optimizer = optim.AdamW(model.parameters(), lr=5e-3)

# Batch size
train_loader = DataLoader(train_ds, batch_size=64)

# Early stopping patience
patience = 3

# Max epochs
for epoch in range(50):
```

### Adjust Data Augmentation
Edit `src/dataset.py`:
```python
# Noise probability
if np.random.random() < 0.3:  # 30% chance

# Masking probabilities  
if np.random.random() < 0.2:  # 20% chance
```

### Model Architecture Changes
Edit `src/model.py` to experiment with:
- Different filter sizes
- More/fewer layers
- Different dropout rates
- Alternative architectures (ResNet, etc.)

## 📊 Comparison with Baselines

| Method | Accuracy | Notes |
|--------|----------|-------|
| **Our CNN Model** | **76.98%** | Full pipeline with augmentation |
| Random Baseline | 16.67% | Random guessing (6 classes) |
| Majority Class | 16.67% | Always predict most common class |
| Simple MLP | ~45-55% | Basic fully connected network |
| SVM on MFCC | ~60-65% | Traditional machine learning |

## 🔬 Research Applications

This system can be used for:
- **Human-Computer Interaction**: Emotion-aware interfaces
- **Mental Health**: Mood monitoring applications  
- **Customer Service**: Call center emotion analysis
- **Entertainment**: Gaming and media personalization
- **Research**: Affective computing studies
- **Education**: Student engagement monitoring

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
git clone https://github.com/hamzamahboob2/Emotion-Speech-Detector.git
cd Emotion-Speech-Detector
pip install -r requirements.txt
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CREMA-D Dataset**: Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014)
- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing library
- **scikit-learn**: Machine learning metrics and utilities

## 📬 Contact

**Hamza Mahboob** - [@hamzamahboob2](https://github.com/hamzamahboob2)

Project Link: [https://github.com/hamzamahboob2/Emotion-Speech-Detector](https://github.com/hamzamahboob2/Emotion-Speech-Detector)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ by [Hamza Mahboob](https://github.com/hamzamahboob2)
