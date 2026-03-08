import pandas as pd
import numpy as np
from pathlib import Path

def process_ingestion_pipeline(data_path):
    """Loads data and decides between training (3 outputs) or prediction (2 outputs)."""
    try:
        df = pd.read_csv(Path(data_path), low_memory=False)
        
        if 'loan_status' in df.columns:
            print("--- Status found: Running Training Flow (3 outputs) ---")
            return _handle_training_data(df)
        else:
            print("--- Status missing: Running Prediction Flow (2 outputs) ---")
            return _handle_prediction_data(df)
            
    except Exception as e:
        print(f"Ingestion Error: {e}")
        return None

def _handle_training_data(df):
    """Func 1: Uses Training Selectors. Returns (df_pd, df_lgd_train, df_lgd_pred)."""
    unique_vals = set(df['loan_status'].dropna().unique())
    is_already_binary = unique_vals.issubset({0, 1, 0.0, 1.0})

    if is_already_binary:
        df['loan_status'] = df['loan_status'].astype(int)
        finished_df = df.copy()
        ongoing_df = pd.DataFrame(columns=df.columns) 
    else:
        finished_statuses = [
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        ]
        ongoing_statuses = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
        
        status_map = {
            'Fully Paid': 0, 'Does not meet the credit policy. Status:Fully Paid': 0,
            'Charged Off': 1, 'Does not meet the credit policy. Status:Charged Off': 1, 'Default': 1
        }
        
        finished_df = df[df['loan_status'].isin(finished_statuses)].copy()
        ongoing_df = df[df['loan_status'].isin(ongoing_statuses)].copy()
        finished_df['loan_status'] = finished_df['loan_status'].map(status_map).astype(int)
    
    # Using specific Training Selectors
    return (
        get_pd_training_data(finished_df), 
        get_lgd_training_data(finished_df), 
        get_lgd_prediction_data_with_status(ongoing_df)
    )

def _handle_prediction_data(df):
    """Func 2: Uses Prediction Selectors. Returns (df_pd, df_lgd_pred)."""
    return (
        get_pd_prediction_data(df), 
        get_lgd_prediction_data_no_status(df)
    )

# --- 1. Training Selectors (Include loan_status) ---

def get_pd_training_data(df):
    cols = [
        'id', 'loan_status', 'loan_amnt', 'term', 'grade', 'emp_title', 'home_ownership',
        'annual_inc', 'verification_status', 'dti', 'earliest_cr_line',
        'inq_last_6mths', 'total_rev_hi_lim', 'acc_open_past_24mths',
        'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
        'mths_since_recent_bc', 'mths_since_recent_inq', 'num_actv_rev_tl',
        'num_tl_op_past_12m'
    ]
    return df[[c for c in cols if c in df.columns]].copy()

def get_lgd_training_data(df):
    cols = [
        'id', 'loan_status', 'int_rate', 'term', 'purpose', 'addr_state', 
        'disbursement_method', 'annual_inc', 'home_ownership', 'loan_amnt',
        'total_rec_prncp', 'recoveries', 'collection_recovery_fee', 
        'total_pymnt', 'last_pymnt_amnt', 'debt_settlement_flag'
    ]
    return df[[c for c in cols if c in df.columns]].copy()

def get_lgd_prediction_data_with_status(df):
    """Exogenous features + status for ongoing loans in training flow."""
    cols = ['id', 'loan_status', 'purpose', 'addr_state', 'disbursement_method', 'term', 'int_rate']
    return df[[c for c in cols if c in df.columns]].copy()

# --- 2. Prediction Selectors (Exclude loan_status) ---

def get_pd_prediction_data(df):
    cols = [
        'id', 'loan_amnt', 'term', 'grade', 'emp_title', 'home_ownership',
        'annual_inc', 'verification_status', 'dti', 'earliest_cr_line',
        'inq_last_6mths', 'total_rev_hi_lim', 'acc_open_past_24mths',
        'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
        'mths_since_recent_bc', 'mths_since_recent_inq', 'num_actv_rev_tl',
        'num_tl_op_past_12m'
    ]
    return df[[c for c in cols if c in df.columns]].copy()

def get_lgd_prediction_data_no_status(df):
    """Exogenous features only for new borrowers."""
    cols = ['id', 'purpose', 'addr_state', 'disbursement_method', 'term', 'int_rate']
    return df[[c for c in cols if c in df.columns]].copy()