import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_markov_simulation(df, horizon=1, prior_json_path=None):
    # ... [Keep path setup and TPM initialization the same] ...
    DATA_DIR = Path(__file__).resolve().parent / "../../../data/models/markov_chains/"
    ARTIFACT_DIR = Path(__file__).resolve().parent / "../../../artifacts/markov_chains/"
    DATA_DIR.mkdir(parents=True, exist_ok=True); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    tpm = np.array([
        [0.95, 0.03, 0.01, 0.01, 0.00], # State 0
        [0.40, 0.40, 0.15, 0.04, 0.01], # State 1
        [0.15, 0.10, 0.50, 0.20, 0.05], # State 2
        [0.05, 0.05, 0.10, 0.40, 0.40], # State 3
        [0.00, 0.00, 0.00, 0.00, 1.00]  # State 4
    ])

    results = []
    for _, row in df.iterrows():
        curr = int(row['status'])
        # Predict the most likely outcome for operational tracking
        nxt = np.argmax(tpm[curr])
        
        results.append({
            'id': row['id'],
            'current_state': curr,
            'predicted_next_state': nxt,
            'probability': tpm[curr][nxt]
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(DATA_DIR / "markov_simulation_raw.csv", index=False)
    return output_df, tpm

def analyze_and_report_risk(csv_path, tpm):
    """
    FIXED: Calculates PD for ALL accounts. 
    Uses Matrix Power (tpm^n) to find the probability of reaching State 4.
    """
    df = pd.read_csv(csv_path)
    report_data = []

    # Calculate 12-month forward PD matrix for more realistic risk (tpm^12)
    # Even if they stay in State 0 today, they have a cumulative risk over time.
    tpm_12 = np.linalg.matrix_power(tpm, 12)

    for _, row in df.iterrows():
        curr = int(row['current_state'])
        
        # 1. Determine Tag based on Current State (Regulatory Bucket)
        if curr == 0: tag = "stable"
        elif curr == 1: tag = "late"
        elif curr == 2: tag = "attention"
        elif curr == 3: tag = "warning"
        else: tag = "default"

        # 2. Calculate PD (Probability of ending in State 4)
        # We look at the probability of reaching the absorbing state from 'curr'
        pd_val = tpm_12[curr, 4]

        report_data.append({
            'id': row['id'],
            'status_tag': tag,
            'cumulative_pd': round(pd_val, 4)
        })

    final_df = pd.DataFrame(report_data)
    save_path = Path(__file__).resolve().parent / "../../../data/models/markov_chains/"
    output_file = save_path / "markov_risk_report.csv"
    final_df.to_csv(output_file, index=False)
    
    return final_df