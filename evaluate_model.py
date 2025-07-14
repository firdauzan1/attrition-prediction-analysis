#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 1. Imports
# ==============================================================================
import numpy as np
import joblib
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2. Constants
# ==============================================================================
MODEL_PATH = 'attrition_model.joblib'
DATA_PATH = 'data/'
OUTPUT_PATH = 'evaluation_results/'

# 3. Functions
# ==============================================================================

def load_artifacts(model_path, data_path):
    """
    Load the model and test data that have been saved.

    Args:
        model_path (str): Path to the .joblib model file.
        data_path (str): Path to the folder containing test data .npy files.

    Returns:
        tuple: Contains model, X_test, and y_test.
    """
    try:
        model = joblib.load(model_path)
        X_test = np.load(data_path + 'X_test_processed.npy', allow_pickle=True)
        y_test = np.load(data_path + 'y_test.npy', allow_pickle=True)
        print("✅ Model and test data loaded successfully.")
        return model, X_test, y_test
    except FileNotFoundError as e:
        print(f"❌ Error: File not found. {e}")
        print("Make sure you have run the 'preprocessing.py' and 'train_model.py' scripts first.")
        return None, None, None

def plot_confusion_matrix(y_test, y_pred, output_path):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stay (No)', 'Attrition (Yes)'],
                yticklabels=['Stay (No)', 'Attrition (Yes)'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    print(f"📊 Confusion matrix saved at '{output_path}confusion_matrix.png'.")
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, output_path):
    """Create and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, 'roc_curve.png'))
    print(f"📈 ROC Curve saved at '{output_path}roc_curve.png'.")
    plt.close()

def evaluate_performance(model, X_test, y_test):
    """
    Perform a complete evaluation and display the results.
    
    Args:
        model: Trained machine learning model.
        X_test (np.array): Test feature data.
        y_test (np.array): Test target data.
    """
    print("\n🔬 Making predictions on test data...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability for positive class (Attrition=1)
    
    print("\n--- Classification Report ---")
    # Precision: How accurate are our positive predictions? (TP / (TP + FP))
    # Recall: How many actual positive cases did we identify? (TP / (TP + FN)) -> Important for attrition!
    # F1-Score: Harmonic mean of Precision and Recall.
    report = classification_report(y_test, y_pred, target_names=['Stay (0)', 'Attrition (1)'])
    print(report)
    
    # ROC-AUC Score: Model's ability to distinguish between classes. Closer to 1 is better.
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("-----------------------------\n")

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Create visualizations
    plot_confusion_matrix(y_test, y_pred, OUTPUT_PATH)
    plot_roc_curve(y_test, y_pred_proba, OUTPUT_PATH)


# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    """
    Main orchestration to run the model evaluation process.
    """
    print("\n--- Starting Step 3: Model Evaluation ---")
    
    # Load model and data
    model, X_test_data, y_test_data = load_artifacts(MODEL_PATH, DATA_PATH)
    
    if model is not None:
        # Evaluate performance
        evaluate_performance(model, X_test_data, y_test_data)
        
        print("\n--- Evaluation Complete ---")
        print(f"Evaluation results and plots saved in folder '{OUTPUT_PATH}'.")