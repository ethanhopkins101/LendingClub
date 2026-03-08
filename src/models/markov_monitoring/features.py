import pandas as pd

def clean_markov_data(df):
    """
    Ensures data integrity for Markov Chain transition modeling.
    Removes records with missing statuses and eliminates duplicate IDs.
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame(columns=['id', 'status'])

        # 1. Drop records where status is None (NaN)
        # This removes loans that are not in our 0, 1, 2, 3 state space
        df_cleaned = df.dropna(subset=['status']).copy()

        # 2. Handle Duplicate IDs
        # We only want the most recent or unique observation per ID
        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['id'], keep='first')
        
        dropped_duplicates = initial_count - len(df_cleaned)
        if dropped_duplicates > 0:
            print(f"Cleaned {dropped_duplicates} duplicate IDs.")

        # 3. Type Enforcement
        # Statuses must be integers to serve as matrix indices
        df_cleaned['status'] = df_cleaned['status'].astype(int)

        print(f"Cleaning complete. {len(df_cleaned)} valid states retained.")
        return df_cleaned.reset_index(drop=True)

    except Exception as e:
        print(f"Error in clean_markov_data: {e}")
        return df
    

def clean_action_data(df):
    """
    Cleaning for Bank Provisioning (Action) reporting.
    Retains financial columns while removing duplicates and invalid entries.
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Integrity Check
        # We drop rows only if the ID or the primary provisioning metric is missing
        df_cleaned = df.dropna(subset=['id', 'out_prncp']).copy()

        # 2. Financial Normalization
        # Ensure 'provision_needed' is non-negative (refunds/credits handled separately)
        df_cleaned['provision_needed'] = df_cleaned['provision_needed'].clip(lower=0)

        # 3. Deduplication
        # Keep the record with the highest outstanding principal if duplicates exist
        # (Conservative approach: Provision for the highest possible exposure)
        df_cleaned = df_cleaned.sort_values('out_prncp', ascending=False)
        df_cleaned = df_cleaned.drop_duplicates(subset=['id'], keep='first')

        print(f"Action cleaning complete. {len(df_cleaned)} financial records verified.")
        return df_cleaned.reset_index(drop=True)

    except Exception as e:
        print(f"Error in clean_action_data: {e}")
        return df