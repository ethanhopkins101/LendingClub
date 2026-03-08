import pandas as pd
import pickle
import json
import numpy as np
from pathlib import Path

def get_pd_and_base_profile(df, bin_path, cal_model_path):
    """Generates PD predictions while forcing 2D array structure."""
    with open(bin_path, 'rb') as f:
        binning_process = pickle.load(f)
    with open(cal_model_path, 'rb') as f:
        calibrated_model = pickle.load(f)

    df_calc = df.copy()
    df_calc['ead'] = (df_calc['lti_ratio'] * df_calc['annual_inc']).round(2)

    # 1. Ensure X is a DataFrame (avoids Series flattening)
    expected_features = list(binning_process.variable_names)
    X = df_calc[expected_features]
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # 2. Transform to WoE
    X_woe = binning_process.transform(X)

    # 3. THE CRITICAL FIX: Force 2D Shape
    # If X_woe is [0, 1, 2...], reshape it to [[0], [1], [2]...]
    X_values = np.array(X_woe)
    if X_values.ndim == 1:
        # If we have multiple features but it flattened, reshape to (-1, num_features)
        # If we have 1 feature, this makes it (6, 1)
        X_values = X_values.reshape(-1, len(expected_features))

    # 4. Predict (Now guaranteed to be 2D)
    pd_probs = calibrated_model.predict_proba(X_values)[:, 1]

    return pd.DataFrame({
        'id': df_calc['id'].values,
        'ead': df_calc['ead'].values,
        'pd': pd_probs,
        'base_interest_rate_pct': df_calc['base_interest_rate_pct'].values
    })

def get_lgd_profile(df, lgd_model_path):
    """Generates LGD predictions using One-Hot Encoding and Schema Alignment."""
    with open(lgd_model_path, 'rb') as f:
        lgd_model = pickle.load(f)
    
    schema_path = Path(lgd_model_path).parent / "lgd_feature_names.json"
    with open(schema_path, 'r') as f:
        lgd_expected_features = json.load(f)

    ids = df['id'].values
    X = df.drop(columns=['id', 'int_rate'], errors='ignore')
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_aligned = X_encoded.reindex(columns=lgd_expected_features, fill_value=0)

    # FIX 4: Ensure LGD input is 2D
    X_lgd_vals = X_aligned.values
    if X_lgd_vals.ndim == 1:
        X_lgd_vals = X_lgd_vals.reshape(1, -1)

    lgd_preds = lgd_model.predict(X_lgd_vals)

    return pd.DataFrame({'id': ids, 'lgd': lgd_preds})

def generate_initial_risk_report(pd_df_input, lgd_df_input, cal_model_path, bin_path, lgd_model_path):
    """Orchestrates PD and LGD profiling and merges results."""
    
    # Safety Check: Ensure inputs are not empty
    if pd_df_input.empty or lgd_df_input.empty:
        raise ValueError("Input dataframes for pricing are empty. Check upstream filtering.")

    pd_df_results = get_pd_and_base_profile(pd_df_input, bin_path, cal_model_path)
    lgd_df_results = get_lgd_profile(lgd_df_input, lgd_model_path)

    final_risk_df = pd.merge(pd_df_results, lgd_df_results, on='id', how='inner')

    # Reorder columns
    cols = ['id', 'pd', 'ead', 'lgd', 'base_interest_rate_pct']
    final_risk_df = final_risk_df[cols]

    # Resolve paths correctly for Docker
    BASE_DIR = Path(__file__).resolve().parent
    SAVE_DIR = (BASE_DIR / "../../../data/models/price_engine/").resolve()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = SAVE_DIR / "initial_risk_report.csv"
    final_risk_df.to_csv(output_path, index=False)
    
    print(f"--- [INITIAL PRICING] Report saved to: {output_path}")
    return final_risk_df