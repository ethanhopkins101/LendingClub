import optuna
import json
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import pickle
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



def optimize_pd_models(X_train, X_test, y_train, y_test):
    """
    Runs Optuna studies. Dropping IDs here to ensure zero leakage during tuning.
    """
    try:
        # 1. Protection Layer: Strip non-feature columns
        X_tr = X_train.drop(columns=['id', 'loan_status'], errors='ignore')
        X_te = X_test.drop(columns=['id', 'loan_status'], errors='ignore')

        # Calculate Class Imbalance for XGBoost
        counts = y_train.value_counts()
        scale_ratio = counts[0] / counts[1]

        # --- 1. Logistic Regression Objective ---
        def objective_logit(trial):
            params = {
                'C': trial.suggest_float('C', 1e-4, 10, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'random_state': 42,
                'max_iter': 1000
            }
            model = LogisticRegression(**params)
            model.fit(X_tr, y_train.values.ravel())
            preds = model.predict_proba(X_te)[:, 1]
            return average_precision_score(y_test, preds)

        # --- 2. XGBoost Objective ---
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': scale_ratio,
                'random_state': 42,
                'tree_method': 'hist'
            }
            model = XGBClassifier(**params)
            model.fit(X_tr, y_train)
            preds = model.predict_proba(X_te)[:, 1]
            return average_precision_score(y_test, preds)

        study_logit = optuna.create_study(direction='maximize')
        study_logit.optimize(objective_logit, n_trials=15)

        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(objective_xgb, n_trials=15)

        # Save Best Parameters
        best_params = {
            "logit": study_logit.best_params,
            "xgb": study_xgb.best_params
        }
        best_params["xgb"].update({"scale_pos_weight": scale_ratio, "random_state": 42, "tree_method": "hist"})
        best_params["logit"].update({"solver": "liblinear", "class_weight": "balanced", "random_state": 42})

        JSON_DIR = (Path(__file__).resolve().parent / "../../../json_files/probability_of_default/").resolve()
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(JSON_DIR / "best_pd_params.json", "w") as f:
            json.dump(best_params, f, indent=4)

        print(f"Optimization complete. Features used: {list(X_tr.columns)}")
        return best_params

    except Exception as e:
        print(f"Error in optimization: {e}")
        return None

def train_optimized_models(X, y, json_path):
    """
    Trains final models using the pure feature set (no IDs).
    """
    try:
        # 1. Protection Layer
        X_clean = X.drop(columns=['id', 'loan_status'], errors='ignore')

        with open(json_path, 'r') as f:
            params = json.load(f)
        
        logit_model = LogisticRegression(**params['logit'])
        xgb_model = XGBClassifier(**params['xgb'])
        
        print(f"Training final models on {X_clean.shape[1]} features...")
        logit_model.fit(X_clean, y.values.ravel())
        xgb_model.fit(X_clean, y)
        
        return logit_model, xgb_model
    except Exception as e:
        print(f"Error training: {e}")
        return None, None

def calibrate_and_save_models(logit_raw, xgb_raw, X_val, y_val):
    """
    Calibrates using features only.
    """
    try:
        # 1. Protection Layer
        X_v = X_val.drop(columns=['id', 'loan_status'], errors='ignore')

        cal_logit = CalibratedClassifierCV(logit_raw, method='isotonic', cv='prefit')
        cal_logit.fit(X_v, y_val)
        
        cal_xgb = CalibratedClassifierCV(xgb_raw, method='isotonic', cv='prefit')
        cal_xgb.fit(X_v, y_val)
        
        PKL_DIR = (Path(__file__).resolve().parent / "../../../models/probability_of_default/").resolve()
        PKL_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(PKL_DIR / "logit_pd_calibrated.pkl", "wb") as f:
            pickle.dump(cal_logit, f)
        with open(PKL_DIR / "xgb_pd_calibrated.pkl", "wb") as f:
            pickle.dump(cal_xgb, f)
            
        return cal_logit, cal_xgb
    except Exception as e:
        print(f"Error in calibration: {e}")
        return None, None
    


def optimize_lgd_model(X_train_raw, X_test_raw, y_train, y_test):
    """
    Optimizes LGD RF model using ONLY specified categorical features.
    """
    try:
        # 1. Strict Feature Selection
        target_features = ['purpose', 'addr_state', 'disbursement_method', 'term']
        
        def prepare_lgd_features(df):
            # Select only the 4 requested features
            X = df[target_features].copy()
            # Convert to dummies (Categorical to Numeric)
            X = pd.get_dummies(X, drop_first=True)
            return X

        X_train = prepare_lgd_features(X_train_raw)
        X_test = prepare_lgd_features(X_test_raw)
        
        # Align columns (ensure test set matches train set dummy columns)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # 2. Optimization Objective
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train.values.ravel() if hasattr(y_train, 'values') else y_train)
            preds = model.predict(X_test)
            return mean_squared_error(y_test, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=15)

        # 3. Save Params
        JSON_DIR = (Path(__file__).resolve().parent / "../../../json_files/probability_of_default/").resolve()
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(JSON_DIR / "best_lgd_params.json", 'w') as f:
            json.dump(study.best_params, f, indent=4)

        print(f"LGD Optimization complete. Best MSE: {study.best_value:.4f}")
        return study.best_params

    except Exception as e:
        print(f"Error in LGD optimization: {e}")
        return None

def train_final_lgd_model(df_realized, json_params_path):
    """
    Trains LGD model on full data using only: purpose, addr_state, disbursement_method, term.
    """
    try:
        with open(json_params_path, 'r') as f:
            best_params = json.load(f)

        # 1. Explicit Feature/Target Split
        target_features = ['purpose', 'addr_state', 'disbursement_method', 'term']
        X = df_realized[target_features].copy()
        y = df_realized['lgd']

        # 2. Encode and Fit
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        print(f"Training Final LGD Model. Features: {X_encoded.shape[1]}")
        final_model = RandomForestRegressor(**best_params, random_state=42)
        final_model.fit(X_encoded, y.values.ravel())

        # 3. Save Model and Feature Names
        MODEL_DIR = (Path(__file__).resolve().parent / "../../../models/probability_of_default/").resolve()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(MODEL_DIR / "final_lgd_model.pkl", 'wb') as f:
            pickle.dump(final_model, f)

        with open(MODEL_DIR / "lgd_feature_names.json", 'w') as f:
            json.dump(list(X_encoded.columns), f)

        print(f"LGD Model saved. No IDs or Interest Rates used in training.")
        return final_model

    except Exception as e:
        print(f"Critical error in LGD Training: {e}")
        return None
