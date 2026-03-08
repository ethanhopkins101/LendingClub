import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import norm

def calculate_portfolio_rwa(df, tpm_path):
    try:
        JSON_DIR = Path(__file__).resolve().parent / "../../../json_files/monte_carlo/"
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(tpm_path, 'r') as f:
            tpm = np.array(json.load(f)['tpm'])
        
        tpm_12 = np.linalg.matrix_power(tpm, 12)
        df['pd'] = df['current_state'].apply(lambda x: tpm_12[x, 4]).clip(1e-6, 0.99)
        df['lgd'] = (1 - (df['recoveries'] / df['funded_amnt'])).fillna(0.45).clip(0.05, 1)
        df['ead'] = df['out_prncp']
        
        df['expected_loss'] = df['pd'] * df['lgd'] * df['ead']
        total_portfolio_el = df['expected_loss'].sum()

        # --- 1. Realistic Standardized Approach ---
        # Under Basel III, past-due or high-risk retail exposures carry a 150% weight.
        def get_standard_weight(row):
            # 'status_tag' check helps identify distressed assets
            if hasattr(row, 'status_tag') and row['status_tag'] in ['warning', 'attention']:
                return 1.50
            return 0.75 # Default retail weight

        df['weight'] = df.apply(get_standard_weight, axis=1)
        rwa_standardized = (df['ead'] * df['weight']).sum()

        # --- 2. Foundation IRB (with Regulatory Scaling) ---
        def get_basel_R(pd):
            return 0.03 * (1 - np.exp(-35 * pd)) / (1 - np.exp(-35)) + \
                   0.16 * (1 - (1 - np.exp(-35 * pd)) / (1 - np.exp(-35)))

        df['R'] = df['pd'].apply(get_basel_R)
        
        def get_irb_k(row):
            pd, lgd, R = row['pd'], row['lgd'], row['R']
            Z = (norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)
            return (lgd * norm.cdf(Z) - lgd * pd) * 1.06

        df['K'] = df.apply(get_irb_k, axis=1)
        
        # Scaling factor (0.85) is used to align internal models with regulatory floors
        rwa_irb = (df['K'] * 12.5 * df['ead']).sum() * 0.85

        # --- 3. Monte Carlo (Stochastic Vasicek) ---
        n_sims = 10000
        z_systematic = np.random.standard_normal(n_sims)
        
        thresholds = norm.ppf(df['pd'].values)
        sqrt_R = np.sqrt(df['R'].values)
        sqrt_1_R = np.sqrt(1 - df['R'].values)
        ead_lgd = (df['ead'] * df['lgd']).values
        max_possible_loss = df['ead'].sum()

        portfolio_losses = []
        for i in range(n_sims):
            conditional_pd = norm.cdf((thresholds - sqrt_R * z_systematic[i]) / sqrt_1_R)
            sim_loss = (conditional_pd * ead_lgd).sum()
            portfolio_losses.append(np.minimum(sim_loss, max_possible_loss))

        var_999 = np.percentile(portfolio_losses, 99.9)
        unexpected_loss = max(0, var_999 - total_portfolio_el)
        rwa_monte_carlo = unexpected_loss * 12.5

        results = {
            "rwa_report": {
                "standardized_approach": round(rwa_standardized, 2),
                "irb_formula_approach": round(rwa_irb, 2),
                "monte_carlo_stochastic_approach": round(rwa_monte_carlo, 2),
                "portfolio_expected_loss": round(total_portfolio_el, 2),
                "unit": "USD",
                "simulations_performed": n_sims
            }
        }

        with open(JSON_DIR / "portfolio_rwa_comparison.json", 'w') as f:
            json.dump(results, f, indent=4)

        return results

    except Exception as e:
        print(f"Error in RWA Simulation: {e}")
        return None