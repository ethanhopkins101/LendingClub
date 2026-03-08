import pandas as pd
import numpy as np

def _func1_process_training_data(df):
    """Processes historical data by mapping loan status and filtering ongoing loans."""
    finished_statuses = [
        'Fully Paid', 'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    ]

    status_map = {
        'Fully Paid': 0, 
        'Does not meet the credit policy. Status:Fully Paid': 0,
        'Charged Off': 1, 
        'Does not meet the credit policy. Status:Charged Off': 1, 
        'Default': 1
    }

    # Filter for completed credit events
    df = df[df['loan_status'].isin(finished_statuses)].copy()

    # Convert status to binary int
    try:
        df['loan_status'] = df['loan_status'].astype(int)
    except (ValueError, TypeError):
        df['loan_status'] = df['loan_status'].map(status_map)

    pd_cols = [
        'loan_status', 'loan_amnt', 'term', 'grade', 'emp_title', 'home_ownership',
        'annual_inc', 'verification_status', 'dti', 'earliest_cr_line',
        'inq_last_6mths', 'total_rev_hi_lim', 'acc_open_past_24mths',
        'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
        'mths_since_recent_bc', 'mths_since_recent_inq', 'num_actv_rev_tl',
        'num_tl_op_past_12m', 'purpose'
    ]
    
    # Ensure all columns exist before subsetting
    available_cols = [c for c in pd_cols if c in df.columns]
    pd_train = df[available_cols].copy()
    
    return pd_train

def _func2_process_prediction_data(df):
    """Splits live data into PD and LGD feature sets for inference."""
    pd_cols = [
        'id', 'loan_amnt', 'term', 'grade', 'emp_title', 'home_ownership',
        'annual_inc', 'verification_status', 'dti', 'earliest_cr_line',
        'inq_last_6mths', 'total_rev_hi_lim', 'acc_open_past_24mths',
        'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
        'mths_since_recent_bc', 'mths_since_recent_inq', 'num_actv_rev_tl',
        'num_tl_op_past_12m', 'purpose'
    ]
    
    lgd_cols = ['id', 'purpose', 'addr_state', 'disbursement_method', 'term']

    pd_pred = df[[c for c in pd_cols if c in df.columns]].copy()
    lgd_pred = df[[c for c in lgd_cols if c in df.columns]].copy()

    return pd_pred, lgd_pred

def load_and_route_data(file_path):
    """Main entry point: loads data and routes based on presence of 'loan_status'."""
    # Load data (assuming CSV or Parquet depending on your project needs)
    if str(file_path).endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)

    if 'loan_status' in df.columns:
        print("Routing to Training Pipeline (func1)...")
        return _func1_process_training_data(df)
    else:
        print("Routing to Prediction Pipeline (func2)...")
        return _func2_process_prediction_data(df)