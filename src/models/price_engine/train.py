import json
import pickle
import optuna
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

def optimize_and_save_params(X_train, X_test, y_train, y_test):
    """Uses Optuna to find best Logistic Regression hyperparameters and saves to JSON."""
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-4, 10, log=True),
            'solver': 'liblinear',
            'class_weight': 'balanced',
            'random_state': 42
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)
        return log_loss(y_test, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    JSON_DIR = Path(__file__).resolve().parent / "../../../json_files/price_engine/"
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(JSON_DIR / "best_params.json", 'w') as f:
        json.dump(study.best_params, f)
    
    return study.best_params

def train_final_logit(df, json_path, bin_pkl_path):
    """Trains the Logistic Regression on the full dataset using optimized params."""
    with open(json_path, 'r') as f:
        best_params = json.load(f)
    
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # We assume X is already WoE-binned from the features.py step
    model = LogisticRegression(**best_params, class_weight='balanced')
    model.fit(X, y)
    
    MODEL_DIR = Path(__file__).resolve().parent / "../../../models/price_engine/"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MODEL_DIR / "logit_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    return model

def calibrate_and_save(df, bin_pkl_path, logit_pkl_path):
    """Calibrates the model probabilities using the Isotonic method."""
    with open(logit_pkl_path, 'rb') as f:
        base_model = pickle.load(f)
    
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # cv="prefit" allows us to calibrate an already trained model
    calibrator = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrator.fit(X, y)
    
    MODEL_DIR = Path(__file__).resolve().parent / "../../../models/price_engine/"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(MODEL_DIR / "calibrated_model.pkl", 'wb') as f:
        pickle.dump(calibrator, f)
    
    return calibrator