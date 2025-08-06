import torch
import numpy as np
from model import SimpleCNN
from dataset import CREMADataset, emotion_map
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path='../models/emotion_model_best.pth', dataset_path='../data/processed'):
    """Evaluate the trained model"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    model = SimpleCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Trying alternative path...")
        model.load_state_dict(torch.load('../models/emotion_model.pth', map_location=device))
        print("Loaded model from: ../models/emotion_model.pth")
    
    model = model.to(device)
    model.eval()
    
    # Load test data (clean dataset without augmentation)
    dataset = CREMADataset(dataset_path, augment=False)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Evaluation
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert numeric labels back to emotion names
    emotion_names = list(emotion_map.keys())
    label_names = [emotion_names[i] for i in all_labels]
    pred_names = [emotion_names[i] for i in all_predictions]
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print("=" * 60)
    print(classification_report(label_names, pred_names, target_names=emotion_names))
    
    # Create and display confusion matrix
    cm = confusion_matrix(label_names, pred_names, labels=emotion_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title(f'Confusion Matrix - Overall Accuracy: {accuracy:.4f}', fontsize=14)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.tight_layout()
    plt.savefig('../models/confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: ../models/confusion_matrix_improved.png")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i, emotion in enumerate(emotion_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_predictions)[class_mask] == i)
            print(f"{emotion}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Sample predictions with confidence
    print("\nSample Predictions with Confidence:")
    print("-" * 50)
    for i in range(min(10, len(all_predictions))):
        true_emotion = emotion_names[all_labels[i]]
        pred_emotion = emotion_names[all_predictions[i]]
        confidence = np.max(all_probabilities[i])
        print(f"True: {true_emotion:3s} | Pred: {pred_emotion:3s} | Confidence: {confidence:.3f}")
    
    return accuracy, all_predictions, all_labels

if __name__ == "__main__":
    accuracy, predictions, labels = evaluate_model()
    plt.show()
