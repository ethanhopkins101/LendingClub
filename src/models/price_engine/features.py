import pandas as pd
import numpy as np
import inspect
import sklearn.utils.validation
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from optbinning import BinningProcess
import json


# --- 1. THE REINFORCED PATCH ---
# Resolves version conflicts between OptBinning and newer Scikit-Learn
if not hasattr(sklearn.utils.validation, 'original_check_array'):
    original_check_array = sklearn.utils.validation.check_array
    
    def universal_patch(array, **kwargs):
        sig = inspect.signature(original_check_array)
        # Swap parameter names based on what the current sklearn version expects
        if 'force_all_finite' in kwargs and 'ensure_all_finite' in sig.parameters:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        elif 'ensure_all_finite' in kwargs and 'force_all_finite' in sig.parameters:
            kwargs['force_all_finite'] = kwargs.pop('ensure_all_finite')
            
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return original_check_array(array, **filtered_kwargs)
    
    sklearn.utils.validation.check_array = universal_patch
    sklearn.utils.check_array = universal_patch

def handle_missing_values(df):
    try:
        if df is None or df.empty:
            return df

        df = df.replace(r'^\s*$', np.nan, regex=True)
        # Ensure ID is treated as a string or int, not checked for nullity in the drop
        crucial_cols = ['loan_amnt', 'annual_inc', 'dti', 'loan_status']
        
        # Categorical cleaning: ignore 'id' even if it's an object
        obj_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'id']
        df = df.dropna(subset=[c for c in obj_cols if c in df.columns]).copy()

        num_cols = df.select_dtypes(include=[np.number]).columns
        if df[num_cols].isna().any().any():
            for col in num_cols:
                nan_pct = df[col].isna().mean()
                if nan_pct == 0: continue
                if col in crucial_cols or nan_pct >= 0.05:
                    df = df.dropna(subset=[col])
                else:
                    imputer = SimpleImputer(strategy='median')
                    df.loc[:, [col]] = imputer.fit_transform(df[[col]])
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Error in handle_missing_values: {e}")
        return df

def engineer_features(df):
    try:
        fe_df = df.copy()
        fe_df['lti_ratio'] = fe_df['loan_amnt'] / (fe_df['annual_inc'] + 1)
        
        fe_df['earliest_cr_line'] = pd.to_datetime(fe_df['earliest_cr_line'], errors='coerce')
        # Fill missing dates with a sensible default before calculation
        fe_df['earliest_cr_line'] = fe_df['earliest_cr_line'].fillna(pd.Timestamp('2010-01-01'))
        
        reference_date = pd.Timestamp('2019-01-01')
        fe_df['credit_age_years'] = (reference_date - fe_df['earliest_cr_line']).dt.days / 365.25
        
        fe_df['total_utilization'] = fe_df['avg_cur_bal'] / (fe_df['total_rev_hi_lim'] + 1)
        fe_df['credit_hunger'] = fe_df['inq_last_6mths'] + fe_df['num_tl_op_past_12m']

        # Safe Term Extraction
        if fe_df['term'].dtype == object:
            fe_df['term_num'] = fe_df['term'].str.extract('(\d+)').astype(float)
        else:
            fe_df['term_num'] = fe_df['term'].astype(float)
        
        fe_df['term_num'] = fe_df['term_num'].fillna(36) # Default 36 months
        fe_df['monthly_installment_ratio'] = (fe_df['loan_amnt'] / fe_df['term_num']) / (fe_df['annual_inc'] / 12 + 1)
        fe_df['bc_util_to_open_ratio'] = fe_df['bc_util'] / (fe_df['bc_open_to_buy'] + 1)

        cols_to_drop = ['loan_amnt', 'earliest_cr_line', 'avg_cur_bal', 'inq_last_6mths', 
                        'num_tl_op_past_12m', 'mo_sin_old_rev_tl_op', 'term']
        fe_df = fe_df.drop(columns=[c for c in cols_to_drop if c in fe_df.columns])
        return fe_df
    except Exception as e:
        print(f"Error in engineer_features: {e}")
        return df
    

def sync_dataframes_by_id(df1, df2):
    """
    Synchronizes two dataframes based on the intersection of their 'id' column.
    Ensures that both dataframes contain the exact same borrowers in the same order.
    """
    try:
        # 1. Identify common IDs
        common_ids = np.intersect1d(df1['id'], df2['id'])
        
        # 2. Filter both dataframes to only include the intersection
        # Use .set_index and .loc to ensure alignment
        df1_synced = df1[df1['id'].isin(common_ids)].sort_values('id').reset_index(drop=True)
        df2_synced = df2[df2['id'].isin(common_ids)].sort_values('id').reset_index(drop=True)
        
        print(f"Synchronization complete. Common IDs found: {len(common_ids)}")
        return df1_synced, df2_synced

    except KeyError:
        print("Error: One or both dataframes are missing the 'id' column.")
        return df1, df2
    except Exception as e:
        print(f"Error in sync_dataframes_by_id: {e}")
        return df1, df2
    

class WoETransformer(BaseEstimator, TransformerMixin):
    """Transformer that wraps the OptBinning BinningProcess for scikit-learn pipelines."""
    def __init__(self, binning_process):
        self.binning_process = binning_process
    
    def fit(self, X, y):
        # Convert y to values to ensure compatibility with optbinning internals
        self.binning_process.fit(X, y.values if hasattr(y, 'values') else y)
        return self
    
    def transform(self, X):
        # Transform returns the WoE values; we force float64 for mathematical stability
        X_woe = self.binning_process.transform(X, metric="woe")
        return np.array(X_woe, dtype=float)

def prepare_and_bin_data(df, mode='train'):
    try:
        # Define paths
        MODEL_DIR = Path(__file__).resolve().parent / "../../../models/price_engine/"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        train_path = MODEL_DIR / "binning_train.pkl"
        predict_path = MODEL_DIR / "binning_predict.pkl"

        # 1. Manage IDs
        df_ids = df[['id']].reset_index(drop=True) if 'id' in df.columns else pd.DataFrame({'id': range(len(df))})
        X = df.drop(columns=['id'], errors='ignore').replace([np.inf, -np.inf], np.nan)

        # 2. Handle Target
        y = X.pop('loan_status').astype(int) if 'loan_status' in X.columns else None
        x_cols = X.columns.tolist()

        if mode == 'train':
            bin_proc = BinningProcess(variable_names=x_cols)
            bin_proc.fit(X, y)
            with open(train_path, 'wb') as f:
                pickle.dump(bin_proc, f)
        else:
            # PREDICTION MODE: Load the training bins to ensure consistency
            if not train_path.exists():
                raise FileNotFoundError("Training binning model not found. Run training first.")
            with open(train_path, 'rb') as f:
                bin_proc = pickle.load(f)
            
            # Save a copy as binning_predict.pkl for your audit trail
            with open(predict_path, 'wb') as f:
                pickle.dump(bin_proc, f)

        # 3. Transform
        X_woe_array = bin_proc.transform(X, metric="woe")
        X_woe = pd.DataFrame(X_woe_array, columns=x_cols, index=df.index)

        return df_ids, X_woe, y
    except Exception as e:
        print(f"Error in prepare_and_bin_data: {e}")
        return None, None, None
    
    
    
    except Exception as e:
        print(f"Error in prepare_and_bin_data: {e}")
        return None, None, None
def split_training_data(X, y, test_size=0.2):
    """Splits data into train/test sets with stratification for binary targets."""
    try:
        # Use stratification only for PD (classification) where y is binary
        stratify_param = y if (len(np.unique(y)) <= 2) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=stratify_param
        )
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error in split_training_data: {e}")
        return None, None, None, None
    
def calculate_base_rates(df, json_path=None):
    """
    Calculates the base interest rate per borrower based on loan purpose and macro constants.
    Returns the original dataframe with the 'base_interest_rate_pct' column.
    """
    # 1. Load Constants (JSON or Default)
    if json_path and Path(json_path).exists():
        with open(json_path, 'r') as f:
            config = json.load(f)
    else:
        # Fallback Default Config (Real World 2026 Macro Environment)
        config = {
            "macro_constants": {
                "fed_funds_rate": 0.0525,
                "operational_cost": 0.0200
            },
            "purpose_spreads": {
                "debt_consolidation": 0.0150, "credit_card": 0.0140,
                "home_improvement": 0.0120, "car": 0.0100,
                "major_purchase": 0.0160, "medical": 0.0130,
                "house": 0.0150, "vacation": 0.0250,
                "wedding": 0.0200, "small_business": 0.0350,
                "moving": 0.0250, "renewable_energy": 0.0120,
                "other": 0.0250
            }
        }

    # 2. Extract Constants
    fed_rate = config['macro_constants']['fed_funds_rate']
    ops_cost = config['macro_constants']['operational_cost']
    spreads = config['purpose_spreads']

    # 3. Calculation Logic
    # We use .get() with a default of 0.0250 for unknown purposes
    def get_rate(purpose):
        spread = spreads.get(purpose, 0.0250)
        return (fed_rate + ops_cost + spread) * 100

    # 4. Apply and Round
    df = df.copy()
    df['base_interest_rate_pct'] = df['purpose'].apply(get_rate).round(2)

    return df