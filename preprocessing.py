#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 1. Imports
# ==============================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# 2. Constants
# ==============================================================================
FILE_PATH = 'HR_Employee_Attrition.xlsx'  # Change to your file path
TARGET_COLUMN = 'Attrition'
COLUMNS_TO_DROP = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 3. Functions
# ==============================================================================

def load_data(file_path):
    """
    Load data from the given file path.
    Supports Excel files (.xlsx).

    Args:
        file_path (str): Path to the data file.

    Returns:
        pandas.DataFrame: DataFrame containing the data, or None if failed.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Data loaded successfully from '{file_path}'.")
        print(f"Initial data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error occurred while loading data: {e}")
        return None

def run_preprocessing(df):
    """
    Run the complete data preprocessing pipeline.

    Args:
        df (pandas.DataFrame): Raw DataFrame.

    Returns:
        tuple: Contains processed train and test data
               (X_train, X_test, y_train, y_test), as well as the
               fitted preprocessor and label_encoder objects.
    """
    # Step 1: Initial Cleaning
    df_cleaned = df.drop(columns=COLUMNS_TO_DROP)
    print(f"Columns {COLUMNS_TO_DROP} have been dropped.")

    # Step 2: Split Features (X) and Target (y)
    X = df_cleaned.drop(TARGET_COLUMN, axis=1)
    y = df_cleaned[TARGET_COLUMN]

    # Step 3: Encode Target Variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Target variable '{TARGET_COLUMN}' has been encoded. (Yes=1, No=0)")
    
    # Step 4: Identify feature types
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    print(f"Found {len(numerical_features)} numerical features and {len(categorical_features)} categorical features.")

    # Step 5: Create Preprocessing Pipeline with ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )

    # Step 6: Split data into train and test sets (before SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"Data split into 80% train and 20% test with stratification.")

    # Step 7: Apply preprocessor
    # .fit_transform() ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # .transform() on test data to prevent data leakage
    X_test_processed = preprocessor.transform(X_test)
    print("Preprocessing (Scaling & Encoding) applied.")

    # Step 8: Apply SMOTE ONLY on training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print("SMOTE applied to training data to balance classes.")
    
    print("\n--- Preprocessing Summary ---")
    print(f"Shape of X_train after SMOTE: {X_train_resampled.shape}")
    print(f"Shape of y_train after SMOTE: {y_train_resampled.shape}")
    print(f"Shape of processed X_test: {X_test_processed.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    
    return X_train_resampled, y_train_resampled, X_test_processed, y_test, preprocessor, label_encoder

# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    """
    This block will only be executed if the script is run directly.
    """
    import numpy as np
    import os

    print("--- Starting Data Preprocessing ---")
    
    # Create 'data' folder if it doesn't exist to save output arrays
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load data
    df_raw = load_data(FILE_PATH)
    
    if df_raw is not None:
        # Run preprocessing pipeline
        X_train, y_train, X_test, y_test, preprocessor_obj, label_encoder_obj = run_preprocessing(df_raw)
        
        # Save preprocessor and encoder objects
        joblib.dump(preprocessor_obj, 'preprocessor.joblib')
        joblib.dump(label_encoder_obj, 'label_encoder.joblib')
        print("\n✅ 'preprocessor.joblib' and 'label_encoder.joblib' objects have been saved.")
        
        # <<<--- NEW ADDITION HERE --->>>
        # Save processed data arrays
        np.save('data/X_train_resampled.npy', X_train)
        np.save('data/y_train_resampled.npy', y_train)
        np.save('data/X_test_processed.npy', X_test)
        np.save('data/y_test.npy', y_test)
        print("✅ Training and test data (NumPy arrays) have been saved in the 'data/' folder.")
        
        print("\n--- Preprocessing Complete ---")