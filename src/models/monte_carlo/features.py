import pandas as pd
import numpy as np

def clean_monte_carlo_data(df):
    """
    Refines data for stochastic simulation.
    Ensures EAD (Exposure) and LGD (Loss) components are mathematically sound.
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Integrity Check: ID and Current State are non-negotiable
        df_cleaned = df.dropna(subset=['id', 'current_state']).copy()

        # 2. Financial Normalization (Exposure at Default - EAD)
        # Exposure cannot be negative; if missing, we assume 0 to be conservative
        df_cleaned['out_prncp'] = df_cleaned['out_prncp'].fillna(0).clip(lower=0)
        
        # 3. LGD Component Preparation
        # funded_amnt is our denominator for recovery rates. 
        # We filter out loans with 0 funded_amnt to avoid division by zero errors in simulation.
        df_cleaned = df_cleaned[df_cleaned['funded_amnt'] > 0].copy()
        
        # Recoveries cannot exceed the funded amount (legally/logically in most cases)
        df_cleaned['recoveries'] = df_cleaned['recoveries'].fillna(0).clip(lower=0)
        df_cleaned.loc[df_cleaned['recoveries'] > df_cleaned['funded_amnt'], 'recoveries'] = df_cleaned['funded_amnt']

        # 4. Deduplication
        # Like the Action cleaning, we prioritize the record with the highest Exposure (out_prncp)
        df_cleaned = df_cleaned.sort_values('out_prncp', ascending=False)
        df_cleaned = df_cleaned.drop_duplicates(subset=['id'], keep='first')

        print(f"Monte Carlo Feature Cleaning: {len(df_cleaned)} records verified for stochastic modeling.")
        return df_cleaned.reset_index(drop=True)

    except Exception as e:
        print(f"Error in clean_monte_carlo_data: {e}")
        return df