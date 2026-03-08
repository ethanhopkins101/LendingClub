import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

def clean_initial_review_data(df):
    """
    Standardizes numeric features and handles missing values.
    Note: Categorical encoding is handled separately to avoid leakage.
    """
    df = df.copy()
    
    # 1. Basic Hygiene
    df = df.drop_duplicates()

    # 2. Date Feature: Extract Year
    if 'issue_d' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce').dt.year
        # Use a fixed sentinel (like 2026) or median for missing dates
        df['issue_d'] = df['issue_d'].fillna(df['issue_d'].median())

    # 3. Handle DTI Outliers (Clipping)
    # Scientifically better than dropping: we 'cap' extreme values so 
    # the model can still provide a prediction for high-debt applicants.
    if 'dti' in df.columns:
        upper_bound = df['dti'].quantile(0.99) 
        df['dti'] = df['dti'].clip(upper=upper_bound)
        df['dti'] = df['dti'].fillna(df['dti'].median())

    # 4. Handle Missing FICO
    if 'fico_range_high' in df.columns:
        # -1 identifies "No Credit History" as a unique signal for XGBoost
        df['fico_range_high'] = df['fico_range_high'].fillna(-1)

    return df

def split_initial_review_data(df, target_col='loan_status', test_size=0.2, random_state=42):
    """
    Performs stratified split to preserve default rates in training and testing.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"Split Ratios: Train {len(X_train)} | Test {len(X_test)}")
    return X_train, X_test, y_train, y_test

def fit_transform_encoders(X_train, X_test):
    """
    Fits OrdinalEncoder on Training data and applies to both sets.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    cat_cols = ['emp_length', 'title', 'zip_code', 'addr_state']
    
    # handle_unknown='use_encoded_value' ensures production stability
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
    X_test[cat_cols] = encoder.transform(X_test[cat_cols].astype(str))
    
    # Path logic
    model_dir = Path(__file__).resolve().parent / "../../../models/initial_review/"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "ordinal_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
        
    return X_train, X_test

def apply_encoder_to_df(df, encoder_path):
    """
    Applies pre-trained encoder to incoming prediction data.
    """
    df = df.copy()
    cat_cols = ['emp_length', 'title', 'zip_code', 'addr_state']
    
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
        
    df[cat_cols] = encoder.transform(df[cat_cols].astype(str))
    return df