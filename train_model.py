#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 1. Imports
# ==============================================================================
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 2. Constants
# ==============================================================================
DATA_PATH = 'data/'
MODEL_PATH = 'attrition_model.joblib'
RANDOM_STATE = 42

# 3. Functions
# ==============================================================================

def load_processed_data(path):
    """
    Load previously processed training data.

    Args:
        path (str): Path to the folder containing .npy data files.

    Returns:
        tuple: Contains X_train and y_train arrays.
    """
    try:
        X_train = np.load(path + 'X_train_resampled.npy', allow_pickle=True)
        y_train = np.load(path + 'y_train_resampled.npy', allow_pickle=True)
        print("✅ Training data (X_train, y_train) loaded successfully.")
        return X_train, y_train
    except FileNotFoundError:
        print(f"❌ Error: Make sure the training data files exist in the folder '{path}'.")
        print("Run the 'preprocessing.py' script first.")
        return None, None

def train_classifier(X_train, y_train):
    """
    Initialize, train, and return the classification model.

    Args:
        X_train (np.array): Training feature data.
        y_train (np.array): Training target data.

    Returns:
        sklearn.Model: Trained model.
    """
    print("🚀 Starting training of Random Forest Classifier...")
    
    # Initialize model
    # n_estimators: Number of 'trees' in the 'forest'. More is usually better but slower.
    # random_state: Ensures consistent results every run.
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    
    # Train model
    model.fit(X_train, y_train)
    
    print("✅ Model training complete.")
    return model

def save_model(model, path):
    """
    Save the trained model to a file.

    Args:
        model (sklearn.Model): Model to be saved.
        path (str): File path to save the model.
    """
    joblib.dump(model, path)
    print(f"💾 Model saved at '{path}'.")

# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    """
    Main orchestration to run the model training process.
    """
    print("\n--- Starting Step 2: Model Training ---")
    
    # Load processed data
    X_train_data, y_train_data = load_processed_data(DATA_PATH)
    
    if X_train_data is not None and y_train_data is not None:
        # Train model
        trained_model = train_classifier(X_train_data, y_train_data)
        
        # Save model
        save_model(trained_model, MODEL_PATH)
        
        print("\n--- Model Training Successful ---")
        print(f"Model is ready for evaluation and saved at '{MODEL_PATH}'.")