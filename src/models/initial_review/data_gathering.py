import pandas as pd
from pathlib import Path

# MASTER ORDER: Critical for XGBoost consistency
MASTER_FEATURES = [
    'loan_amnt', 'emp_length', 'issue_d', 'title', 
    'zip_code', 'addr_state', 'dti', 'fico_range_high', 'policy_code'
]

def load_initial_review_data(data_path):
    """
    Objective: Fetches data and enforces a consistent column order.
    Logic: Splits into Training (binary mapping) or Prediction (feature-only).
    """
    path = Path(data_path)
    
    # 1. Peek at columns
    sample = pd.read_csv(path, nrows=0)
    cols = sample.columns.tolist()
    
    # --- TRAINING LOGIC ---
    if 'loan_status' in cols and len(cols) > 9:
        print(f"--- Detected Training Data: {path.name} ---")
        
        # Load features + target
        df = pd.read_csv(path, usecols=MASTER_FEATURES + ['loan_status'])

        # 2. Target Mapping
        finished_statuses = [
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        ]
        status_map = {
            'Fully Paid': 0, 'Does not meet the credit policy. Status:Fully Paid': 0,
            'Charged Off': 1, 'Does not meet the credit policy. Status:Charged Off': 1, 
            'Default': 1
        }

        df = df[df['loan_status'].isin(finished_statuses)].copy()
        df['loan_status'] = df['loan_status'].map(status_map).astype(int)
        
        # Force order: Features first, Target last
        df = df[MASTER_FEATURES + ['loan_status']]
        
        print(f"Binarization Complete. Kept {len(df)} rows.")
        return df
    
    # --- PREDICTION LOGIC ---
    elif len(cols) == 9 and 'loan_status' not in cols:
        print(f"--- Detected Prediction Data: {path.name} ---")
        df = pd.read_csv(path)
        
        # Force the exact Master order to avoid 'feature_names mismatch'
        df = df[MASTER_FEATURES]
        return df
    
    # --- FALLBACK ---
    else:
        print(f"Warning: Unexpected format in {path.name}. Attempting order enforcement.")
        df = pd.read_csv(path)
        valid_cols = [c for c in MASTER_FEATURES if c in df.columns]
        return df[valid_cols]