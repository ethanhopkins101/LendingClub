# import pandas as pd
# from pathlib import Path

# # Importing your modules
# from data_gathering import load_and_map_loan_statuses
# from features import clean_markov_data, clean_action_data
# from simulations import run_markov_simulation, analyze_and_report_risk
# from action import generate_bank_action_reports

# def run_markov_monitoring_pipeline(input_data_path):
#     try:
#         print("--- STARTING MARKOV MONITORING PIPELINE ---")
#         BASE_DATA_DIR = Path(__file__).resolve().parent / "../../../data/models/markov_chains/"
        
#         # --- STEP 1: PREDICTION ---
#         raw_predict_df = load_and_map_loan_statuses(input_data_path, process='predict')
#         cleaned_markov = clean_markov_data(raw_predict_df)
#         sim_results, tpm_matrix = run_markov_simulation(cleaned_markov, horizon=1)
        
#         raw_sim_path = BASE_DATA_DIR / "markov_simulation_raw.csv"
#         risk_report_df = analyze_and_report_risk(raw_sim_path, tpm_matrix)
#         final_sim_path = BASE_DATA_DIR / "markov_risk_report.csv"

#         # --- STEP 2: ACTION ---
#         print("\n[Phase 2: Bank Action & Provisioning]")
#         raw_action_df = load_and_map_loan_statuses(input_data_path, process='action')
#         cleaned_action = clean_action_data(raw_action_df)
        
#         # This now returns the DataFrame (ensure action.py returns detailed_report_df)
#         action_report_df = generate_bank_action_reports(cleaned_action, final_sim_path)

#         print("\n--- PIPELINE EXECUTION COMPLETE ---")
        
#         # Check if the report exists and is a DataFrame before checking status_tag
#         if isinstance(action_report_df, pd.DataFrame) and 'status_tag' in action_report_df.columns:
#             risk_count = len(action_report_df[action_report_df['status_tag'] != 'stable'])
#             print(f"Total Loans Monitored: {len(action_report_df)}")
#             print(f"Risk Actions identified: {risk_count}")
#         else:
#             print("Pipeline finished. Summary metrics saved to JSON.")

#         return action_report_df

#     except Exception as e:
#         print(f"CRITICAL PIPELINE ERROR: {e}")
#         return None

# if __name__ == "__main__":
#     DATA_PATH = Path(__file__).resolve().parent / "../../../data/cleaned/markovian/ongoing_loans.csv"
#     run_markov_monitoring_pipeline(DATA_PATH)

import pandas as pd
from pathlib import Path
import sys


# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/models/markov_monitoring/execution.py -> Root is 3 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../../").resolve()

# Ensure the project root and src directory are in sys.path for absolute imports
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Importing modules using absolute project paths
from src.models.markov_monitoring.data_gathering import load_and_map_loan_statuses
from src.models.markov_monitoring.features import clean_markov_data, clean_action_data
from src.models.markov_monitoring.simulations import run_markov_simulation, analyze_and_report_risk
from src.models.markov_monitoring.action import generate_bank_action_reports

def run_markov_monitoring_pipeline(input_data_path):
    """
    Executes the Markov monitoring pipeline.
    Saves results to data/models/markov_chains/
    """
    try:
        print("--- STARTING MARKOV MONITORING PIPELINE ---")
        
        # Define absolute paths relative to root for consistency inside Docker
        BASE_DATA_DIR = ROOT / "data" / "models" / "markov_chains"
        BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Resolve the input data path relative to ROOT if not already absolute
        target_input = Path(input_data_path)
        abs_input_path = target_input if target_input.is_absolute() else (ROOT / input_data_path).resolve()

        if not abs_input_path.exists():
            raise FileNotFoundError(f"Markov input data not found at: {abs_input_path}")

        # --- STEP 1: PREDICTION ---
        raw_predict_df = load_and_map_loan_statuses(str(abs_input_path), process='predict')
        cleaned_markov = clean_markov_data(raw_predict_df)
        sim_results, tpm_matrix = run_markov_simulation(cleaned_markov, horizon=1)
        
        raw_sim_path = BASE_DATA_DIR / "markov_simulation_raw.csv"
        # Passing absolute string path to the analysis utility
        risk_report_df = analyze_and_report_risk(str(raw_sim_path), tpm_matrix)
        
        final_sim_path = BASE_DATA_DIR / "markov_risk_report.csv"

        # --- STEP 2: ACTION ---
        print("\n[Phase 2: Bank Action & Provisioning]")
        raw_action_df = load_and_map_loan_statuses(str(abs_input_path), process='action')
        cleaned_action = clean_action_data(raw_action_df)
        
        # Generate report and save it to the path the frontend expects
        action_report_df = generate_bank_action_reports(cleaned_action, str(final_sim_path))
        
        # Save the final detailed report specifically where the frontend retrieves it
        OUTPUT_REPORT = BASE_DATA_DIR / "detailed_action_report.csv"
        action_report_df.to_csv(OUTPUT_REPORT, index=False)

        print(f"\n✅ REPORT SAVED TO: {OUTPUT_REPORT}")
        print("--- PIPELINE EXECUTION COMPLETE ---")
        
        if isinstance(action_report_df, pd.DataFrame) and 'status_tag' in action_report_df.columns:
            risk_count = len(action_report_df[action_report_df['status_tag'] != 'stable'])
            print(f"Total Loans Monitored: {len(action_report_df)}")
            print(f"Risk Actions identified: {risk_count}")
        
        return action_report_df

    except Exception as e:
        print(f"CRITICAL PIPELINE ERROR: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input ongoing loans CSV")
    args = parser.parse_args()

    # Use provided data or fall back to default path relative to ROOT
    data_path = args.data if args.data else str(ROOT / "data" / "generated" / "markovian_sample.csv")
    run_markov_monitoring_pipeline(data_path)