import xgboost as xgb
import optuna
import json
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV


def optimize_initial_review_xgb(X_train, X_test, y_train, y_test):
    """
    Objective 1: Use Optuna to find best hyperparameters and save to JSON.
    """
    # Calculate scale_pos_weight for imbalance
    counter = y_train.value_counts()
    ratio = counter[0] / counter[1]

    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'scale_pos_weight': ratio,
            'tree_method': 'hist',
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.7, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
        }
        
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # We optimize for Average Precision (PR-AUC)
        preds_probs = model.predict_proba(X_test)[:, 1]
        return average_precision_score(y_test, preds_probs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)

    # Save Best Params
    out_path = Path(__file__).resolve().parent / "../../../json_files/initial_review/best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"Optimization complete. Params saved to: {out_path}")
    return study.best_params

def train_and_save_final_model(df, json_params_path):
    """
    Objective 2: Train final model on full data using best params and save as pkl.
    """
    # 1. Load Params
    with open(json_params_path, 'r') as f:
        best_params = json.load(f)
    
    # 2. Prepare Data (X and y)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Calculate ratio for the final run
    counter = y.value_counts()
    ratio = counter[0] / counter[1]
    
    # 3. Add required static params
    best_params.update({
        'scale_pos_weight': ratio,
        'tree_method': 'hist',
        'n_estimators': 1000
    })
    
    # 4. Train
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X, y)
    
    # 5. Save Model
    model_out = Path(__file__).resolve().parent / "../../../models/initial_review/xgb_initial_review.pkl"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_out, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"Final model trained and saved to: {model_out}")



def train_and_save_calibrated_model(df, model_pkl_path):
    """
    Objective: Calibrate the pre-trained XGBoost model to provide 
    reliable probability estimates.
    """
    # 1. Load the Base XGBoost Model
    with open(model_pkl_path, 'rb') as f:
        base_xgb = pickle.load(f)
        
    # 2. Prepare Data
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # 3. Initialize Calibrated Model
    # 'isotonic' is non-parametric and works best with large samples (>1000)
    # cv='prefit' tells sklearn the model is already trained
    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb, 
        method='isotonic', 
        cv='prefit'
    )
    
    # 4. Fit Calibration
    print("Calibrating model probabilities...")
    calibrated_model.fit(X, y)
    
    # 5. Save Calibrated Model
    out_dir = Path(__file__).resolve().parent / "../../../models/initial_review/"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "calibrated_xgb_initial_review.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(calibrated_model, f)
        
    print(f"Calibrated model saved to: {out_path}")