import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import sklearn.utils.validation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from optbinning import BinningProcess
import pickle
from pathlib import Path
import inspect

# --- THE REINFORCED PATCH (Fixes 'ensure_all_finite' error) ---
if not hasattr(sklearn.utils.validation, 'original_check_array'):
    original_check_array = sklearn.utils.validation.check_array
    
    def universal_patch(array, **kwargs):
        # Get the arguments that the actual sklearn function accepts
        sig = inspect.signature(original_check_array)
        
        # Handle the Optbinning/Sklearn version naming conflict
        if 'force_all_finite' in kwargs and 'ensure_all_finite' in sig.parameters:
            kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
        elif 'ensure_all_finite' in kwargs and 'force_all_finite' in sig.parameters:
            kwargs['force_all_finite'] = kwargs.pop('ensure_all_finite')
            
        # Only pass kwargs that exist in the original function signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return original_check_array(array, **filtered_kwargs)
    
    sklearn.utils.validation.check_array = universal_patch
    sklearn.utils.check_array = universal_patch

def handle_missing_values(df):
    try:
        if df is None or df.empty:
            return df

        # 1. CRITICAL: Convert empty strings or whitespace-only strings to NaN
        # This fixes: "could not convert string to float: ''"
        df = df.replace(r'^\s*$', np.nan, regex=True)

        crucial_cols = ['loan_amnt', 'annual_inc', 'dti', 'loan_status']
        obj_cols = df.select_dtypes(include=['object']).columns
        
        # Drop rows where categorical data is missing
        df = df.dropna(subset=[c for c in obj_cols if c in df.columns]).copy()

        num_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check if we actually need to impute anything
        if df[num_cols].isna().any().any():
            for col in num_cols:
                nan_pct = df[col].isna().mean()
                if nan_pct == 0:
                    continue
                
                # Drop crucial or high-nullity columns
                if col in crucial_cols or nan_pct >= 0.05:
                    df = df.dropna(subset=[col])
                else:
                    # Impute low-nullity columns
                    imputer = IterativeImputer(
                        estimator=RandomForestRegressor(n_estimators=10, n_jobs=-1), 
                        max_iter=5, # Reduced iterations for speed
                        random_state=42
                    )
                    # fit_transform expects numeric only
                    imputed_values = imputer.fit_transform(df[num_cols])
                    col_idx = list(num_cols).index(col)
                    
                    # Fix: Use .loc to avoid SettingWithCopyWarning
                    df.loc[:, col] = imputed_values[:, col_idx]

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Error in handle_missing_values: {e}")
        return df

def engineer_features(df):
    try:
        fe_df = df.copy()
        fe_df['lti_ratio'] = fe_df['loan_amnt'] / (fe_df['annual_inc'] + 1)

        fe_df['earliest_cr_line'] = pd.to_datetime(fe_df['earliest_cr_line'], errors='coerce')
        reference_date = pd.Timestamp('2019-01-01')
        fe_df['credit_age_years'] = (reference_date - fe_df['earliest_cr_line']).dt.days / 365.25
        fe_df['total_utilization'] = fe_df['avg_cur_bal'] / (fe_df['total_rev_hi_lim'] + 1)
        fe_df['credit_hunger'] = fe_df['inq_last_6mths'] + fe_df['num_tl_op_past_12m']

        # Fix: Corrected Regex from (\2+) to (\d+)
        fe_df['term_num'] = fe_df['term'].str.extract('(\d+)').astype(float)
        fe_df['monthly_installment_ratio'] = (fe_df['loan_amnt'] / fe_df['term_num']) / (fe_df['annual_inc'] / 12 + 1)
        fe_df['bc_util_to_open_ratio'] = fe_df['bc_util'] / (fe_df['bc_open_to_buy'] + 1)

        cols_to_drop = ['loan_amnt', 'earliest_cr_line', 'avg_cur_bal', 'inq_last_6mths', 
                        'num_tl_op_past_12m', 'mo_sin_old_rev_tl_op', 'term']
        existing_drops = [c for c in cols_to_drop if c in fe_df.columns]
        fe_df = fe_df.drop(columns=existing_drops)
        return fe_df
    except Exception as e:
        print(f"Error in engineer_features: {e}")
        return df

def sync_model_datasets_train(df_pd, df_lgd_train, df_lgd_pred):
    """
    Used during Training: Syncs the 3 dataframes from the training flow.
    - Syncs df_pd and df_lgd_train (Finished Loans) to match exactly.
    - Ensures df_lgd_pred (Ongoing Loans) is formatted correctly.
    """
    try:
        # 1. Force ID to string across all three
        for df in [df_pd, df_lgd_train, df_lgd_pred]:
            if df is not None and 'id' in df.columns:
                df['id'] = df['id'].astype(str)

        # 2. Sync the Training Pair (Finished Loans)
        if df_pd is not None and df_lgd_train is not None:
            common_train_ids = set(df_pd['id']).intersection(set(df_lgd_train['id']))
            
            df_pd_synced = (df_pd[df_pd['id'].isin(common_train_ids)]
                            .sort_values('id').reset_index(drop=True))
            df_lgd_train_synced = (df_lgd_train[df_lgd_train['id'].isin(common_train_ids)]
                                   .sort_values('id').reset_index(drop=True))
        else:
            df_pd_synced, df_lgd_train_synced = df_pd, df_lgd_train

        # 3. Clean up the Prediction set (Ongoing Loans)
        df_lgd_pred_synced = df_lgd_pred
        if df_lgd_pred is not None:
            df_lgd_pred_synced = df_lgd_pred.sort_values('id').reset_index(drop=True)

        print(f"Training Sync complete. PD/LGD Train rows: {len(df_pd_synced)}. LGD Pred rows: {len(df_lgd_pred_synced or [])}")
        return df_pd_synced, df_lgd_train_synced, df_lgd_pred_synced
        
    except Exception as e:
        print(f"Critical error in sync_model_datasets_train: {e}")
        return df_pd, df_lgd_train, df_lgd_pred
    

def sync_model_datasets_predict(df_pd, df_exog):
    """
    Used during Prediction: Syncs PD features with LGD exogenous features.
    Crucial for re-attaching IDs correctly after the scorecard runs.
    """
    if df_pd is None or df_exog is None:
        print("Prediction Sync skipped: One of the datasets is None.")
        return df_pd, df_exog
    
    try:
        # 1. Force ID to string
        df_pd['id'] = df_pd['id'].astype(str)
        df_exog['id'] = df_exog['id'].astype(str)

        # 2. Sync: Only keep records present in both (pruning rows dropped in PD cleaning)
        surviving_ids = df_pd['id'].unique()
        df_exog_synced = df_exog[df_exog['id'].isin(surviving_ids)].copy()
        
        # 3. Final Sort alignment
        df_pd_synced = df_pd.sort_values('id').reset_index(drop=True)
        df_exog_synced = df_exog_synced.sort_values('id').reset_index(drop=True)
        
        print(f"Prediction Sync complete. Ready for scoring: {len(df_pd_synced)} records.")
        return df_pd_synced, df_exog_synced

    except Exception as e:
        print(f"Critical error in sync_model_datasets_predict: {e}")
        return df_pd, df_exog

# --- WoE Components ---
class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, binning_process):
        self.binning_process = binning_process
    def fit(self, X, y):
        self.binning_process.fit(X, y.values if hasattr(y, 'values') else y)
        return self
    def transform(self, X):
        # transform returns a numpy array
        X_woe = self.binning_process.transform(X, metric="woe")
        # CRITICAL: Force the array to float64 to catch empty strings here
        return np.array(X_woe, dtype=float)
    
def prepare_and_bin_data(df):
    try:
        if 'id' in df.columns:
            df_ids = df[['id']].reset_index(drop=True)
            df = df.drop(columns=['id'])
        else:
            df_ids = pd.DataFrame({'id': range(len(df))})

        df = df.replace([np.inf, -np.inf], np.nan)
        X = df.drop(columns=['loan_status'])
        y = df['loan_status'].astype(int)
        x_cols = X.columns.tolist()

        bin_proc = BinningProcess(variable_names=x_cols)
        woe_pipeline = Pipeline([('woe_transform', WoETransformer(bin_proc))])
        X_woe_array = woe_pipeline.fit_transform(X, y)
        X_woe = pd.DataFrame(X_woe_array, columns=x_cols, index=df.index)

        BASE_DIR = Path(__file__).resolve().parent
        MODEL_DIR = BASE_DIR / "../../../models/probability_of_default/"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "binning_process.pkl", 'wb') as f:
            pickle.dump(bin_proc, f)

        return df_ids, X_woe, y
    except Exception as e:
        print(f"Error in prepare_and_bin_data: {e}")
        return None, None, None

def split_training_data(X, y, target='loan_status', test_size=0.2):
    try:
        # If y is passed as a string (the target name), extract it from X
        if isinstance(y, str):
            target_name = y
            y = X[target_name]
            X = X.drop(columns=[target_name])
            
        # Use stratification only for classification (PD), not for regression (LGD)
        stratify_param = y if (y.nunique() <= 2) else None
        
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_param)
    except Exception as e:
        print(f"Error in split_training_data: {e}")
        return None, None, None, None

def calculate_realized_lgd(df_lgd_raw):
    try:
        # Crucial Fix: LGD calculation requires loan_amnt. Ensure it exists.
        if 'loan_amnt' not in df_lgd_raw.columns:
            print("Error: 'loan_amnt' missing from LGD input. LGD cannot be calculated.")
            return None

        df_defaults = df_lgd_raw[df_lgd_raw['loan_status'].isin(['Charged Off', 1, 'Default'])].copy()
        if df_defaults.empty: return pd.DataFrame()

        df_defaults['EAD'] = df_defaults['loan_amnt'] - df_defaults['total_rec_prncp']
        df_defaults['Net_Rec'] = df_defaults['recoveries'] - df_defaults['collection_recovery_fee']
        df_defaults['lgd'] = np.where(df_defaults['EAD'] > 0, 1 - (df_defaults['Net_Rec'] / df_defaults['EAD']), 0)
        df_defaults['lgd'] = df_defaults['lgd'].clip(0, 1)

        final_cols = ['id', 'purpose', 'addr_state', 'disbursement_method', 'term', 'int_rate', 'lgd']
        existing_cols = [c for c in final_cols if c in df_defaults.columns]
        return df_defaults[existing_cols].reset_index(drop=True)
    except Exception as e:
        print(f"Error in calculate_realized_lgd: {e}")
        return None