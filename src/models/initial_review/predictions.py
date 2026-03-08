import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np

def generate_initial_review_predictions(df, model_pkl_path, threshold=0.5):
    """
    Objective: Uses the trained XGBoost model to predict defaults on new data.
    
    Args:
        df (pd.DataFrame): The feature-engineered dataframe for prediction.
        model_pkl_path (str/Path): Path to the saved .pkl model.
        threshold (float): Probability threshold for classifying a 'Default' (1).
        
    Returns:
        pd.DataFrame: A dataframe containing the binary predictions.
    """
    try:
        # 1. Load the trained model
        with open(model_pkl_path, 'rb') as f:
            model = pickle.load(f)
        
        # 2. Obtain Probabilities
        # predict_proba returns [prob_0, prob_1]
        probs = model.predict_proba(df)[:, 1]
        
        # 3. Apply Threshold for Binary Classification
        # If prob > threshold, label as 1 (Default), else 0
        predictions = (probs >= threshold).astype(int)
        
        # 4. Create Output DataFrame
        # We preserve the original index to match the input data
        results_df = pd.DataFrame(
            {'initial_prediction': predictions}, 
            index=df.index
        )
        
        # 5. Define Output Path and Save
        output_dir = Path(__file__).resolve().parent / "../../../data/models/initial_review/"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "initial_review_results.csv"
        results_df.to_csv(output_file)
        
        print(f"Predictions saved successfully to: {output_file}")
        print(f"Total Rejections (1s): {predictions.sum()} out of {len(predictions)}")
        
        return results_df

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    

def generate_strategy_analysis(df, calibrated_model_path):
    """
    Objective: Create a strategy table showing the impact of 100 
    probability thresholds on approvals and error types with population percentages.
    """
    # 1. Load Calibrated Model
    with open(calibrated_model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 2. Prepare Data
    X = df.drop(columns=['loan_status'], errors='ignore')
    y_true = df['loan_status']
    total_population = len(df)
    
    # 3. Get Calibrated Probabilities
    y_probs = model.predict_proba(X)[:, 1]
    
    # 4. Generate Strategy Metrics
    thresholds = np.linspace(0, 1, 100)
    stats = []

    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        approval_percentage = (y_pred_t == 0).mean() * 100
        
        stats.append({
            'Threshold': round(t, 4),
            'Approval %': round(approval_percentage, 2),
            'FP Count': fp,
            'FN Count': fn,
            'FP % of Total': round((fp / total_population) * 100, 2),
            'FN % of Total': round((fn / total_population) * 100, 2),
            'TP Count': tp,
            'TN Count': tn
        })

    # 5. Save results to CSV (Specific Absolute Path Resolution)
    strategy_df = pd.DataFrame(stats)
    
    # Resolves to project root from src/models/initial_review/ and targets data folder
    output_dir = (Path(__file__).resolve().parent / "../../../data/models/initial_review/").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "strategy_analysis_report.csv"
    strategy_df.to_csv(output_path, index=False)
    
    print(f"Strategy analysis saved to: {output_path}")
    return strategy_df