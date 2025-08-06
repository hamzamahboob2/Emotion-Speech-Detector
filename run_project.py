"""
Complete Emotion Detection Project Runner
This script runs the entire pipeline: preprocessing, training, evaluation, and inference
"""

import os
import sys
import subprocess
import time

def run_command(command, description, working_dir=None):
    """Run a command and display its output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        if working_dir:
            result = subprocess.run(command, shell=True, cwd=working_dir, 
                                  capture_output=False, text=True, check=True)
        else:
            result = subprocess.run(command, shell=True, capture_output=False, 
                                  text=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} not found: {file_path}")
        return False

def main():
    """Run the complete emotion detection pipeline"""
    print("🎵 EMOTION DETECTION PROJECT PIPELINE 🎵")
    print("=========================================")
    
    # Define paths - Fixed path resolution
    project_root = r"c:\Users\shahz\Desktop\Deep Learning emotional detection"
    src_dir = os.path.join(project_root, 'src')
    models_dir = os.path.join(project_root, 'models')
    data_processed = os.path.join(project_root, 'data', 'processed')
    
    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    
    # Create necessary directories
    os.makedirs(models_dir, exist_ok=True)
    
    # Step 1: Check if preprocessing is needed
    print(f"\n📊 STEP 1: DATA PREPROCESSING")
    print("-" * 40)
    
    if not os.path.exists(data_processed) or len(os.listdir(data_processed)) == 0:
        print("Processed data not found. Running preprocessing...")
        if not run_command("C:/Python312/python.exe preprocessing.py", 
                          "Data Preprocessing", src_dir):
            return False
    else:
        print(f"✅ Processed data already exists: {len(os.listdir(data_processed))} files")
    
    # Step 2: Training
    print(f"\n🧠 STEP 2: MODEL TRAINING")
    print("-" * 40)
    
    best_model_path = os.path.join(models_dir, 'emotion_model_best.pth')
    
    if not os.path.exists(best_model_path):
        print("Trained model not found. Starting training...")
        if not run_command("C:/Python312/python.exe train.py", 
                          "Model Training", src_dir):
            return False
    else:
        print("✅ Trained model already exists")
        user_input = input("Do you want to retrain the model? (y/n): ").lower()
        if user_input == 'y':
            if not run_command("C:/Python312/python.exe train.py", 
                              "Model Retraining", src_dir):
                return False
    
    # Step 3: Model Evaluation
    print(f"\n📈 STEP 3: MODEL EVALUATION")
    print("-" * 40)
    
    if not run_command("C:/Python312/python.exe evaluate.py", 
                      "Model Evaluation", src_dir):
        print("⚠️ Evaluation failed, but continuing...")
    
    # Step 4: Inference Demo
    print(f"\n🎯 STEP 4: INFERENCE DEMO")
    print("-" * 40)
    
    if not run_command("C:/Python312/python.exe inference.py", 
                      "Inference Demo", src_dir):
        print("⚠️ Inference demo failed, but pipeline completed")
    
    # Step 5: Project Summary
    print(f"\n📋 STEP 5: PROJECT SUMMARY")
    print("-" * 40)
    
    print("\n🎉 PROJECT PIPELINE COMPLETED! 🎉")
    print("\nGenerated Files:")
    
    # Check for generated files
    files_to_check = [
        (os.path.join(models_dir, 'emotion_model_best.pth'), "Best trained model"),
        (os.path.join(models_dir, 'emotion_model_final.pth'), "Final trained model"),
        (os.path.join(models_dir, 'confusion_matrix_improved.png'), "Confusion matrix"),
        (os.path.join(data_processed), f"Processed data ({len(os.listdir(data_processed)) if os.path.exists(data_processed) else 0} files)")
    ]
    
    for file_path, description in files_to_check:
        check_file_exists(file_path, description)
    
    print(f"\n📚 Next Steps:")
    print("1. Check the confusion matrix: models/confusion_matrix_improved.png")
    print("2. Use inference.py to test on new audio files")
    print("3. Modify model architecture in model.py for better performance")
    print("4. Adjust hyperparameters in train.py")
    print("5. Add more data augmentation techniques")
    
    print(f"\n🏆 Your emotion detection model is ready to use!")

if __name__ == "__main__":
    main()
