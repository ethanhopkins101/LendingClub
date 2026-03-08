import pandas as pd
import numpy as np
from pathlib import Path

def load_and_map_loan_statuses(data_path, process='predict'):
    """
    Fetches loan data for Markov Chain modeling or Bank Provisioning.
    
    Arguments:
        process (str): 'predict' for Markov state mapping, 
                      'action' for Bank Provisioning (includes LGD components).
    """
    try:
        abs_path = Path(data_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Data not found at {abs_path}")
            
        status_map = {
            "Current": 0,
            "In Grace Period": 1,
            "Late (16-30 days)": 2,
            "Late (31-120 days)": 3
        }

        if process == 'predict':
            df = pd.read_csv(abs_path, usecols=['id', 'loan_status'])
            df['status'] = df['loan_status'].map(status_map)
            result_df = df[['id', 'status']].copy()
            
            print(f"Prediction Data: {result_df['status'].notna().sum()} active states identified.")
            return result_df

        elif process == 'action':
            # Scientific Requirement: To calculate LGD and ECL, we need:
            # 1. out_prncp: The Exposure at Default (EAD)
            # 2. funded_amnt & recoveries: Components for Loss Given Default (LGD)
            action_cols = [
                'id', 'loan_status', 'out_prncp', 'funded_amnt', 
                'recoveries', 'total_pymnt'
            ]
            
            df = pd.read_csv(abs_path, usecols=action_cols)
            
            # Map status for context and define provisioning need
            df['status_code'] = df['loan_status'].map(status_map)
            df['provision_needed'] = df['out_prncp'].fillna(0).round(2)
            
            print(f"Action Data: Loaded {len(df)} records with LGD components.")
            return df

    except Exception as e:
        print(f"Error in Markov data_gathering ({process}): {e}")
        return pd.DataFrame()