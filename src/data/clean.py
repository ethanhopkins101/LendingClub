import pandas as pd
import numpy as np
import re


def initial_clean(df):
    """
    Initial data cleaning and validation for the Pricing Engine funnel.
    """
    try:
        # 1. Ensure the data received isn't empty or full of NA
        if df is None or df.empty:
            raise ValueError("Dataframe is empty. No data received at server.")
        
        if df.isna().all().all():
            raise ValueError("Dataframe contains only NaN values.")

        print(f"Initial Shape: {df.shape}")

        # 2. Drop duplicated rows
        try:
            df = df.drop_duplicates()
        except Exception as e:
            print(f"Warning: Could not drop duplicates. Error: {e}")

        # 3. Drop rows if NaN presence exceeds 50% of variables
        try:
            threshold_rows = df.shape[1] * 0.5
            df = df.dropna(thresh=int(threshold_rows), axis=0)
        except Exception as e:
            print(f"Warning: Could not perform row-wise NaN dropping. Error: {e}")

        # 4. Define crucial columns that must NOT be dropped regardless of NaN count
        crucial_cols = [
            'id', 'loan_amnt', 'term', 'emp_length', 'home_ownership', 
            'annual_inc', 'purpose', 'addr_state', 'dti', 'fico_range_low', 
            'fico_range_high', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util'
        ]

        # 5. Drop columns if NaN exceeds 25%, unless they are in crucial_cols
        # Handles missing columns like 'loan_status' or 'int_rate' by checking if they exist in df.columns
        try:
            limit_pct = 0.25
            cols_to_keep = []
            
            for col in df.columns:
                # If it's a target or crucial, we keep it if it exists
                if col in ['loan_status', 'int_rate'] or col in crucial_cols:
                    cols_to_keep.append(col)
                    continue
                
                # For all other columns, check NaN percentage
                nan_pct = df[col].isna().mean()
                if nan_pct <= limit_pct:
                    cols_to_keep.append(col)
            
            df = df[cols_to_keep]
        except Exception as e:
            print(f"Warning: Could not perform column-wise NaN dropping. Error: {e}")
        return df
    except Exception as e:
        print(f"Critical Error in initial cleaning stage: {e}")

# Complete Type Mapping for the 151 LendingClub Columns
    type_map = {
        'id': 'object', 'member_id': 'object', 'loan_amnt': 'float64', 'funded_amnt': 'float64', 
        'funded_amnt_inv': 'float64', 'term': 'object', 'int_rate': 'float64', 'installment': 'float64', 
        'grade': 'object', 'sub_grade': 'object', 'emp_title': 'object', 'emp_length': 'object', 
        'home_ownership': 'object', 'annual_inc': 'float64', 'verification_status': 'object', 
        'issue_d': 'object', 'loan_status': 'object', 'pymnt_plan': 'object', 'url': 'object', 
        'desc': 'object', 'purpose': 'object', 'title': 'object', 'zip_code': 'object', 
        'addr_state': 'object', 'dti': 'float64', 'delinq_2yrs': 'float64', 'earliest_cr_line': 'object', 
        'fico_range_low': 'float64', 'fico_range_high': 'float64', 'inq_last_6mths': 'float64', 
        'mths_since_last_delinq': 'float64', 'mths_since_last_record': 'float64', 'open_acc': 'float64', 
        'pub_rec': 'float64', 'revol_bal': 'float64', 'revol_util': 'float64', 'total_acc': 'float64', 
        'initial_list_status': 'object', 'out_prncp': 'float64', 'out_prncp_inv': 'float64', 
        'total_pymnt': 'float64', 'total_pymnt_inv': 'float64', 'total_rec_prncp': 'float64', 
        'total_rec_int': 'float64', 'total_rec_late_fee': 'float64', 'recoveries': 'float64', 
        'collection_recovery_fee': 'float64', 'last_pymnt_d': 'object', 'last_pymnt_amnt': 'float64', 
        'next_pymnt_d': 'object', 'last_credit_pull_d': 'object', 'last_fico_range_high': 'float64', 
        'last_fico_range_low': 'float64', 'collections_12_mths_ex_med': 'float64', 
        'mths_since_last_major_derog': 'float64', 'policy_code': 'float64', 'application_type': 'object', 
        'annual_inc_joint': 'float64', 'dti_joint': 'float64', 'verification_status_joint': 'object', 
        'acc_now_delinq': 'float64', 'tot_coll_amt': 'float64', 'tot_cur_bal': 'float64', 
        'open_acc_6m': 'float64', 'open_act_il': 'float64', 'open_il_12m': 'float64', 
        'open_il_24m': 'float64', 'mths_since_rcnt_il': 'float64', 'total_bal_il': 'float64', 
        'il_util': 'float64', 'open_rv_12m': 'float64', 'open_rv_24m': 'float64', 
        'max_bal_bc': 'float64', 'all_util': 'float64', 'total_rev_hi_lim': 'float64', 
        'inq_fi': 'float64', 'total_cu_tl': 'float64', 'inq_last_12m': 'float64', 
        'acc_open_past_24mths': 'float64', 'avg_cur_bal': 'float64', 'bc_open_to_buy': 'float64', 
        'bc_util': 'float64', 'chargeoff_within_12_mths': 'float64', 'delinq_amnt': 'float64', 
        'mo_sin_old_il_acct': 'float64', 'mo_sin_old_rev_tl_op': 'float64', 
        'mo_sin_rcnt_rev_tl_op': 'float64', 'mo_sin_rcnt_tl': 'float64', 'mort_acc': 'float64', 
        'mths_since_recent_bc': 'float64', 'mths_since_recent_bc_dlq': 'float64', 
        'mths_since_recent_inq': 'float64', 'mths_since_recent_revol_delinq': 'float64', 
        'num_accts_ever_120_pd': 'float64', 'num_actv_bc_tl': 'float64', 'num_actv_rev_tl': 'float64', 
        'num_bc_sats': 'float64', 'num_bc_tl': 'float64', 'num_il_tl': 'float64', 
        'num_op_rev_tl': 'float64', 'num_rev_accts': 'float64', 'num_rev_tl_bal_gt_0': 'float64', 
        'num_sats': 'float64', 'num_tl_120dpd_2m': 'float64', 'num_tl_30dpd': 'float64', 
        'num_tl_90g_dpd_24m': 'float64', 'num_tl_op_past_12m': 'float64', 'pct_tl_nvr_dlq': 'float64', 
        'percent_bc_gt_75': 'float64', 'pub_rec_bankruptcies': 'float64', 'tax_liens': 'float64', 
        'tot_hi_cred_lim': 'float64', 'total_bal_ex_mort': 'float64', 'total_bc_limit': 'float64', 
        'total_il_high_credit_limit': 'float64', 'revol_bal_joint': 'float64', 
        'sec_app_fico_range_low': 'float64', 'sec_app_fico_range_high': 'float64', 
        'sec_app_earliest_cr_line': 'object', 'sec_app_inq_last_6mths': 'float64', 
        'sec_app_mort_acc': 'float64', 'sec_app_open_acc': 'float64', 'sec_app_revol_util': 'float64', 
        'sec_app_open_act_il': 'float64', 'sec_app_num_rev_accts': 'float64', 
        'sec_app_chargeoff_within_12_mths': 'float64', 'sec_app_collections_12_mths_ex_med': 'float64', 
        'sec_app_mths_since_last_major_derog': 'float64', 'hardship_flag': 'object', 
        'hardship_type': 'object', 'hardship_reason': 'object', 'hardship_status': 'object', 
        'deferral_term': 'float64', 'hardship_amount': 'float64', 'hardship_start_date': 'object', 
        'hardship_end_date': 'object', 'payment_plan_start_date': 'object', 'hardship_length': 'float64', 
        'hardship_dpd': 'float64', 'hardship_loan_status': 'object', 
        'orig_projected_additional_accrued_interest': 'float64', 'hardship_payoff_balance_amount': 'float64', 
        'hardship_last_payment_amount': 'float64', 'disbursement_method': 'object', 
        'debt_settlement_flag': 'object', 'debt_settlement_flag_date': 'object', 
        'settlement_status': 'object', 'settlement_date': 'object', 'settlement_amount': 'float64', 
        'settlement_percentage': 'float64', 'settlement_term': 'float64'
        }

    try:
        # 1. Type Mapping Loop
        # type_map assumed to be defined globally as the 151-column dictionary
        for col, dtype in type_map.items():
            if col in df.columns:
                try:
                    if dtype == 'float64':
                        if df[col].dtype == 'object':
                            df[col] = (df[col].astype(str)
                                    .str.replace('%', '', regex=False)
                                    .str.replace('$', '', regex=False)
                                    .str.replace(',', '', regex=False))
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}. Error: {e}")
    except NameError:
        print("Error: 'type_map' is not defined.")
    except Exception as e:
        print(f"Unexpected error in type mapping: {e}")


def sanitize_data(df):
    """
    Advanced sanitization for numeric and categorical financial data.
    """
    try:
        # 1. Strip spaces and weird characters from Object columns
        try:
            obj_cols = df.select_dtypes(include=['object']).columns
            for col in obj_cols:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'^[,]+|[,]+$', '', regex=True)
                
                # 2. Handle Object-based placeholders
                placeholders = ['?', '-1', '-1.0', 'NAN', 'nan', 'n/a', 'None']
                df[col] = df[col].replace(placeholders, np.nan)
        except Exception as e:
            print(f"Warning: Object sanitization failed. Error: {e}")

        # 3. Sanitize Numeric columns
        try:
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                df[col] = df[col].replace([-1, -1.0], np.nan)
                
                # Logic: No negative values for most financial columns
                if col not in ['dti', 'revol_util', 'dti_joint']:
                    if col in df.columns:
                        df.loc[df[col] < 0, col] = np.nan
        except Exception as e:
            print(f"Warning: Numeric sanitization failed. Error: {e}")

        # 4. Handle 'id' Column
        if 'id' in df.columns:
            try:
                df['id'] = pd.to_numeric(df['id'], errors='coerce')
                df['id'] = df['id'].fillna(0).astype('Int64')
            except Exception as e:
                print(f"Warning: ID column processing failed. Error: {e}")

        # 5. Date Normalization
        date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Date conversion failed for {col}. Error: {e}")

        # 6. Categorical Consistency (term)
        if 'term' in df.columns:
            try:
                df['term'] = df['term'].str.replace(r'\s+', ' ', regex=True).str.strip()
            except Exception as e:
                print(f"Warning: 'term' normalization failed. Error: {e}")

        # 7. Reset Index
        try:
            df = df.reset_index(drop=True)
        except Exception as e:
            print(f"Warning: Index reset failed. Error: {e}")

    except Exception as e:
        print(f"Critical error during sanitize_data: {e}")

    return df

def full_loan_information_processed(df):
    """
    Orchestrator function: Combines initial cleaning and advanced sanitization.
    Ensures structural integrity for both training and inference.
    """
    try:
        # 1. Verification: Ensure input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            print("Error: Input is not a pandas DataFrame.")
            return None

        # 2. Step 1: Initial Cleaning (Dropping NaNs/Duplicates)
        try:
            print("Starting Initial Cleaning...")
            df = initial_clean(df) 
            if df is None:
                raise ValueError("initial_clean returned None")
        except Exception as e:
            print(f"Warning: initial_clean failed. Proceeding with raw data. Error: {e}")

        # 3. Step 2: Advanced Sanitization (Types, Placeholders, Dates)
        try:
            print("Starting Advanced Sanitization...")
            df = sanitize_data(df)
            if df is None:
                raise ValueError("sanitize_data returned None")
        except Exception as e:
            print(f"Warning: sanitize_data failed. Error: {e}")

        # 4. Final Validation and Return
        if df is not None:
            print(f"Full Processing Complete. Final Shape: {df.shape}")
            return df # Move this here to return the cleaned data
        else:
            print("Error: Processing returned None.")
            return None

    except Exception as e:
        print(f"Critical failure in full_loan_information_processed: {e}")
        return None



def clean_initial_review_data(df):
    """
    Cleans the rejection/initial review dataframe and maps names to Master format.
    """
    try:
        # 1. Define and apply Column Mapping
        try:
            rejection_map = {
                'loan_amnt': 'Amount Requested',
                'issue_d': 'Application Date',
                'title': 'Loan Title',
                'fico_range_high': 'Risk_Score',
                'dti': 'Debt-To-Income Ratio',
                'zip_code': 'Zip Code',
                'addr_state': 'State',
                'emp_length': 'Employment Length',
                'policy_code': 'Policy Code'
            }
            # Reverse for mapping rejection names -> master names
            reverse_rejection_map = {v: k for k, v in rejection_map.items()}
            
            # Check if columns are within the map and rename them
            df = df.rename(columns=reverse_rejection_map)
        except Exception as e:
            print(f"Warning: Column mapping failed. Error: {e}")

        if df is None or df.empty:
            return df

        # 2. Basic structural cleaning
        try:
            df = df.drop_duplicates()
            df = df.reset_index(drop=True)
        except Exception as e:
            print(f"Warning: Basic structural cleanup failed: {e}")

        # 3. Iterate through columns and apply sanitization using Master Names
        for col in df.columns:
            try:
                # --- Step A: Universal String Cleaning ---
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].str.replace(r'^[,]+|[,]+$', '', regex=True)
                    
                    placeholders = ['?', '-1', '-1.0', 'NAN', 'nan', 'n/a', 'None']
                    df[col] = df[col].replace(placeholders, np.nan)

                # --- Step B: Type Transformations (Using Master Names) ---
                
                # Numeric Columns
                if col in ['loan_amnt', 'fico_range_high', 'policy_code']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if col != 'policy_code':
                        df.loc[df[col] < 0, col] = np.nan

                # Special Numeric: DTI
                elif col == 'dti':
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.replace('%', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Date Column
                elif col == 'issue_d':
                    df[col] = pd.to_datetime(df[col], errors='coerce')

                # Object Columns (title, zip_code, addr_state, emp_length)
                else:
                    df[col] = df[col].astype('object')

            except Exception as e:
                print(f"Warning: Failed to process column '{col}'. Error: {e}")

        # 4. Final row cleanup
        df = df.dropna(how='all', axis=0).reset_index(drop=True)

    except Exception as e:
        print(f"Critical error in clean_initial_review_data: {e}")

    return df