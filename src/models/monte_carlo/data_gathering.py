import pandas as pd
import numpy as np
from pathlib import Path

def load_monte_carlo_raw_data(file_path):
    """
    Pulls essential columns for Monte Carlo and RWA calculations.
    - Adds 'purpose' for Method 1 (Standardized Approach).
    """
    try:
        abs_path = Path(file_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Data file not found at: {abs_path}")

        # ADDED: 'purpose' to map hard RWA weights
        required_cols = [
            'id', 
            'loan_status', 
            'out_prncp', 
            'funded_amnt', 
            'recoveries',
            'int_rate',
            'installment',
            'purpose' 
        ]

        df = pd.read_csv(abs_path, usecols=required_cols)

        status_map = {
            "Current": 0,
            "In Grace Period": 1,
            "Late (16-30 days)": 2,
            "Late (31-120 days)": 3
        }
        
        df['current_state'] = df['loan_status'].map(status_map)
        df = df.dropna(subset=['current_state']).copy()
        df['current_state'] = df['current_state'].astype(int)

        # Basic financial sanitization
        df['out_prncp'] = df['out_prncp'].fillna(0).clip(lower=0)
        df['funded_amnt'] = df['funded_amnt'].fillna(0).clip(lower=1)
        df['recoveries'] = df['recoveries'].fillna(0).clip(lower=0)
        # Ensure purpose is string for mapping
        df['purpose'] = df['purpose'].fillna('other').astype(str)

        print(f"Monte Carlo Gathering: {len(df)} loans prepared with purpose-mapping capability.")
        return df

    except Exception as e:
        print(f"Error in Monte Carlo data_gathering: {e}")
        return pd.DataFrame()