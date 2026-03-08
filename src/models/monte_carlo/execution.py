# import pandas as pd
# from pathlib import Path

# # Importing your Monte Carlo modules
# from data_gathering import load_monte_carlo_raw_data
# from features import clean_monte_carlo_data
# from simulations import calculate_portfolio_rwa

# def run_monte_carlo_rwa_pipeline(input_data_path):
#     """
#     Executes the Monte Carlo RWA Pipeline:
#     1. Gathers stochastic-relevant features (EAD, LGD components, Purpose).
#     2. Cleans and normalizes financial distributions.
#     3. Simulates 10,000 scenarios to compare Standardized, IRB, and MC RWA.
#     """
#     try:
#         print("--- STARTING MONTE CARLO RWA PIPELINE ---")
        
#         # 1. Define Paths
#         # The TPM is needed to derive the PDs (Probability of Default)
#         TPM_PATH = Path(__file__).resolve().parent / "../../../artifacts/markov_chains/transition_matrix.json"
        
#         # --- STEP 1: DATA GATHERING ---
#         print("\n[Phase 1: Stochastic Data Gathering]")
#         raw_mc_df = load_monte_carlo_raw_data(input_data_path)
        
#         if raw_mc_df.empty:
#             raise ValueError("Data gathering returned an empty DataFrame.")

#         # --- STEP 2: FEATURE CLEANING ---
#         print("\n[Phase 2: Distribution Cleaning]")
#         cleaned_mc_df = clean_monte_carlo_data(raw_mc_df)

#         # --- STEP 3: RWA SIMULATIONS ---
#         print("\n[Phase 3: RWA Comparison Simulation]")
#         # This function performs Method 1 (Standardized), 2 (IRB), and 3 (Monte Carlo)
#         rwa_results = calculate_portfolio_rwa(cleaned_mc_df, TPM_PATH)

#         if rwa_results:
#             print("\n--- PIPELINE EXECUTION COMPLETE ---")
#             report = rwa_results['rwa_report']
#             print(f"Standardized RWA:  ${report['standardized_approach']:,.2f}")
#             print(f"IRB Formula RWA:   ${report['irb_formula_approach']:,.2f}")
#             print(f"Monte Carlo RWA:   ${report['monte_carlo_stochastic_approach']:,.2f}")
        
#         return rwa_results

#     except Exception as e:
#         print(f"CRITICAL MONTE CARLO PIPELINE ERROR: {e}")
#         return None

# if __name__ == "__main__":
#     # Targeting the ongoing loans subset
#     DATA_PATH = Path(__file__).resolve().parent / "../../../data/cleaned/markovian/ongoing_loans.csv"
#     run_monte_carlo_rwa_pipeline(DATA_PATH)


import os
import sys
import pandas as pd
from pathlib import Path
import json

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/models/monte_carlo/execution.py -> Root is 3 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../../").resolve()

# Ensure the project root and src directory are in sys.path for absolute imports
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Absolute imports from the project root
from src.models.monte_carlo.data_gathering import load_monte_carlo_raw_data
from src.models.monte_carlo.features import clean_monte_carlo_data
from src.models.monte_carlo.simulations import calculate_portfolio_rwa



def run_monte_carlo_rwa_pipeline(input_data_path):
    """
    Executes the Monte Carlo RWA Pipeline.
    Saves results to json_files/monte_carlo/portfolio_rwa_comparison.json
    """
    try:
        print("--- STARTING MONTE CARLO RWA PIPELINE ---")
        
        # 1. Define Paths relative to the project root for Docker volume consistency
        TPM_PATH = ROOT / "artifacts" / "markov_chains" / "transition_matrix.json"
        OUTPUT_JSON_PATH = ROOT / "json_files" / "monte_carlo" / "portfolio_rwa_comparison.json"
        
        # Ensure output directory exists inside the container
        OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Resolve the input data path relative to ROOT if not already absolute
        target_input = Path(input_data_path)
        abs_input_path = target_input if target_input.is_absolute() else (ROOT / input_data_path).resolve()

        if not abs_input_path.exists():
            raise FileNotFoundError(f"Monte Carlo input data not found at: {abs_input_path}")

        # --- STEP 1: DATA GATHERING ---
        print(f"\n[Phase 1: Stochastic Data Gathering] Reading: {abs_input_path}")
        raw_mc_df = load_monte_carlo_raw_data(str(abs_input_path))
        
        if raw_mc_df.empty:
            raise ValueError("Data gathering returned an empty DataFrame.")

        # --- STEP 2: FEATURE CLEANING ---
        print("\n[Phase 2: Distribution Cleaning]")
        cleaned_mc_df = clean_monte_carlo_data(raw_mc_df)

        # --- STEP 3: RWA SIMULATIONS ---
        print("\n[Phase 3: RWA Comparison Simulation]")
        # This function performs Method 1 (Standardized), 2 (IRB), and 3 (Monte Carlo)
        # Passing absolute string path for the transition matrix
        rwa_results = calculate_portfolio_rwa(cleaned_mc_df, str(TPM_PATH))

        if rwa_results:
            # Save results to the expected JSON path for the backend to find
            with open(OUTPUT_JSON_PATH, "w") as f:
                json.dump(rwa_results, f, indent=4)
                
            print("\n--- PIPELINE EXECUTION COMPLETE ---")
            report = rwa_results['rwa_report']
            print(f"Standardized RWA:   ${report['standardized_approach']:,.2f}")
            print(f"IRB Formula RWA:    ${report['irb_formula_approach']:,.2f}")
            print(f"Monte Carlo RWA:    ${report['monte_carlo_stochastic_approach']:,.2f}")
            print(f"Results saved to: {OUTPUT_JSON_PATH}")
        
        return rwa_results

    except Exception as e:
        print(f"CRITICAL MONTE CARLO PIPELINE ERROR: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input stochastic data CSV")
    args = parser.parse_args()

    # Use provided data or fall back to default path relative to ROOT
    data_path = args.data if args.data else str(ROOT / "data" / "generated" / "markovian_sample.csv")
    run_monte_carlo_rwa_pipeline(data_path)