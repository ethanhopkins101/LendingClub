import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

def generate_credit_scorecard(df_woe, model_pkl_path, binning_process_path, df_ids):
    """Predicts PD and Scores while maintaining ID alignment."""
    try:
        # 1. Load Model and Sync IDs
        with open(model_pkl_path, 'rb') as f:
            cal_model = pickle.load(f)
        with open(binning_process_path, 'rb') as f:
            binning_process = pickle.load(f)
            
        base_logit = cal_model.estimator if hasattr(cal_model, 'estimator') else cal_model
        
        # Ensure row-order alignment
        df_ids = df_ids.sort_values('id').reset_index(drop=True)
        df_woe = df_woe.sort_index().reset_index(drop=True) # Assumes index was synced to ID

        # 2. Scorecard Scaling Parameters
        pdo, target_score, target_odds = 20, 600, 50
        factor = pdo / np.log(2)
        offset = target_score - (factor * np.log(target_odds))
        intercept = base_logit.intercept_[0]
        coefficients = base_logit.coef_[0]
        
        expected_features = base_logit.feature_names_in_

        # 3. Numeric Enforcement for Matrix Math
        df_woe_numeric = (df_woe.apply(pd.to_numeric, errors='coerce')
                          .reindex(columns=expected_features, fill_value=0.0)
                          .fillna(0.0))

        # 4. Final Scoring Logic
        # Calculate PD using the calibrated model (blind to ID)
        y_probs_cal = cal_model.predict_proba(df_woe_numeric)[:, 1]
        
        # Calculate Credit Score using weighted sum
        weighted_sum = df_woe_numeric.values @ coefficients + intercept
        scores = offset - (weighted_sum * factor)

        # 5. Re-attach IDs
        results_df = pd.DataFrame({
            'id': df_ids['id'].values,
            'probability_of_default': y_probs_cal,
            'credit_score': scores.round().astype(int)
        })

        return results_df

    except Exception as e:
        print(f"Error in scorecard generation: {e}")
        return None


def calculate_expected_loss_and_profit(df_ids, df_binned, df_unbinned, df_exog, pd_model_path, lgd_model_path, binning_process_path):
    """Main Risk Engine: Merges PD and LGD outputs into a final financial report."""
    try:
        # 1. Standardize and Sort (The "Secret Sauce" for alignment)
        for df in [df_ids, df_exog, df_unbinned]:
            df['id'] = df['id'].astype(str)
        
        df_ids = df_ids.sort_values('id').reset_index(drop=True)
        df_exog = df_exog.sort_values('id').reset_index(drop=True)
        df_unbinned = df_unbinned.sort_values('id').reset_index(drop=True)

        # 2. Get PD Results
        results_df = generate_credit_scorecard(df_binned, pd_model_path, binning_process_path, df_ids)
        if results_df is None: return None

        # 3. LGD Prediction (ID-Blind)
        import pickle, json
        with open(lgd_model_path, 'rb') as f:
            lgd_model = pickle.load(f)
        with open(Path(lgd_model_path).parent / "lgd_feature_names.json", 'r') as f:
            lgd_expected_features = json.load(f)

        # Drop ID and int_rate ONLY for the model input, not the dataframe
        lgd_input = df_exog.drop(columns=['id', 'int_rate'], errors='ignore')
        lgd_features = pd.get_dummies(lgd_input, drop_first=True)
        lgd_features = lgd_features.reindex(columns=lgd_expected_features, fill_value=0)
        
        # Add LGD array to results_df (synced by sort order)
        results_df['LGD_pct'] = lgd_model.predict(lgd_features)

        # 4. Financial Calculations
        # We need to pick up int_rate from df_exog and others from df_unbinned
        # Check for existence of columns before slicing to avoid IndexErrors
        needed_cols = ['id', 'lti_ratio', 'annual_inc', 'term_num']
        available_cols = [c for c in needed_cols if c in df_unbinned.columns]
        
        financials = df_unbinned[available_cols].copy()
        
        # Merge PD/LGD results with int_rate (from exog) and other financials
        final_df = results_df.merge(df_exog[['id', 'int_rate']], on='id', how='inner')
        final_df = final_df.merge(financials, on='id', how='inner')

        # Exposure At Default (EAD)
        # Using .get() or fillna to handle missing engineered features safely
        final_df['EAD'] = (final_df['lti_ratio'] * final_df['annual_inc']).round(2)
        
        # Expected Loss: $$EL = PD \times LGD \times EAD$$
        final_df['Expected_Loss'] = (final_df['probability_of_default'] * final_df['LGD_pct'] * final_df['EAD']).round(2)
        
        # Expected Revenue: (EAD * Rate * Time in Years)
        # term_num is months (36 or 60)
        final_df['Expected_Revenue'] = (final_df['EAD'] * (final_df['int_rate'] / 100) * (final_df['term_num'] / 12)).round(2)
        final_df['Expected_Profit'] = (final_df['Expected_Revenue'] - final_df['Expected_Loss']).round(2)

        # 5. Export
        output_cols = ['id', 'probability_of_default', 'credit_score', 'LGD_pct', 'EAD', 'Expected_Loss', 'Expected_Revenue', 'Expected_Profit']
        report = final_df[output_cols].copy()
        
        output_path = Path(__file__).resolve().parent / "../../../data/models/probability_of_default/final_risk_report.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

        print(f"Report Generated: Total Expected Profit: ${report['Expected_Profit'].sum():,.2f}")
        return report

    except Exception as e:
        print(f"Critical error in risk engine: {e}")
        import traceback
        traceback.print_exc()
        return None