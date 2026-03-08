# import pandas as pd
# import sys
# import os
# from pathlib import Path

# # 1. Setup Absolute Paths
# BASE_DIR = Path(__file__).resolve().parent
# SRC_PATH = BASE_DIR / "src"
# DATA_OUTPUT_DIR = BASE_DIR / "data" / "clean"

# # 2. Add src to path for imports
# sys.path.append(str(SRC_PATH))

# try:
#     from data.clean import full_loan_information_processed, clean_initial_review_data
# except ImportError as e:
#     print(f"Critical Import Error: {e}")
#     sys.exit(1)

# def run_full_processing():
#     # Ensure the output directory exists
#     try:
#         DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#         print(f"Output directory verified at: {DATA_OUTPUT_DIR}")
#     except Exception as e:
#         print(f"Error creating directory: {e}")
#         return

#     # Define Input Paths
#     accepted_path = BASE_DIR / 'archive/accepted_2007_to_2018/accepted_2007_to_2018Q4.csv'
#     rejected_path = BASE_DIR / 'archive/rejected_2007_to_2018/rejected_2007_to_2018Q4.csv'

#     # --- STEP 1: Full Accepted Data ---
#     try:
#         print("\n--- Processing Full Accepted Data (This may take several minutes) ---")
#         if accepted_path.exists():
#             # REMOVED nrows=10000 to process the entire file
#             df_accepted = pd.read_csv(accepted_path, low_memory=False)
            
#             df_acc_cleaned = full_loan_information_processed(df_accepted)
            
#             if df_acc_cleaned is not None:
#                 output_file = DATA_OUTPUT_DIR / "accepted_cleaned_full.csv"
#                 df_acc_cleaned.to_csv(output_file, index=False)
#                 print(f"Successfully saved full cleaned accepted data to: {output_file}")
#         else:
#             print(f"Source file not found: {accepted_path}")
#     except Exception as e:
#         print(f"Error processing Full Accepted Data: {e}")

#     # --- STEP 2: Full Rejected Data ---
#     try:
#         print("\n--- Processing Full Rejected Data ---")
#         if rejected_path.exists():
#             # REMOVED nrows=10000 to process the entire file
#             df_rejected = pd.read_csv(rejected_path, low_memory=False)
            
#             df_rej_cleaned = clean_initial_review_data(df_rejected)
            
#             if df_rej_cleaned is not None:
#                 output_file = DATA_OUTPUT_DIR / "rejected_cleaned_full.csv"
#                 df_rej_cleaned.to_csv(output_file, index=False)
#                 print(f"Successfully saved full cleaned rejected data to: {output_file}")
#         else:
#             print(f"Source file not found: {rejected_path}")
#     except Exception as e:
#         print(f"Error processing Full Rejected Data: {e}")

# if __name__ == "__main__":
#     run_full_processing()


# import os
# import sys
# import pandas as pd
# from pathlib import Path

# def test_backend_execution():
#     # 1. SETUP PATHS
#     project_root = Path(__file__).resolve().parent
#     # This points to where your model logic actually sits
#     model_src_path = project_root / "src" / "models" / "initial_review"
    
#     # --- CRITICAL: Add the model folder to sys.path so local imports work ---
#     if str(model_src_path) not in sys.path:
#         sys.path.append(str(model_src_path))
    
#     # Now import the pipe after the path is set
#     from execution import run_full_pipe

#     # We use the sample data which doesn't have labels (loan_status)
#     input_rel_path = "data/generated/sample_data.csv"
#     absolute_input_path = (project_root / input_rel_path).resolve()
#     test_threshold = 0

#     print(f"--- [TEST START: PREDICTION ONLY] ---")
#     print(f"Target Input: {absolute_input_path}")
#     print(f"Threshold:    {test_threshold}")

#     # 2. VALIDATE INPUT
#     if not absolute_input_path.exists():
#         print(f"❌ ERROR: Input file not found at {absolute_input_path}")
#         return

#     # 3. EXECUTE
#     try:
#         # Pass mode="predict" to bypass training logic entirely
#         run_full_pipe(str(absolute_input_path), threshold=test_threshold, mode="predict")
#         print(f"--- [PIPE SUCCESS] ---")
        
#         # 4. VERIFY OUTPUTS
#         # These are usually written to data/models/initial_review/
#         output_dir = project_root / "data" / "models" / "initial_review"
#         expected = ["initial_review_results.csv", "strategy_analysis_report.csv"]
        
#         print("\nChecking for results in:", output_dir)
#         for file in expected:
#             file_path = output_dir / file
#             status = "✅ FOUND" if file_path.exists() else "❌ MISSING"
#             print(f"{file:35} : {status}")
            
#     except Exception as e:
#         print(f"--- [PIPE FAILED] ---")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     test_backend_execution()


import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. ENVIRONMENT & PATH SETUP ---
# Resolves to the project root
BASE_DIR = Path(__file__).resolve().parent

# Force project root into sys.path so 'src' is discoverable
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# --- 2. IMPORT PRICING LOGIC ---
try:
    from src.models.price_engine.execution import run_pricing_pipeline
    print("✅ Successfully imported Price Engine logic.")
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    print("Ensure you are running this from the LendingClub root directory.")
    sys.exit(1)

# --- 3. PATH CONFIGURATION ---
DATA_IN = BASE_DIR / "data" / "generated" / "risk_engine_sample_generated.csv"
MODEL_DIR = BASE_DIR / "models" / "price_engine"

def main():
    print("--- 🚀 STARTING PRICE ENGINE TEST ---")
    
    # 1. Check Input Data
    if not DATA_IN.exists():
        print(f"❌ ERROR: Input file not found at {DATA_IN}")
        print("Please ensure the sample data generator has been run.")
        return

    # 2. Check for Required Artifacts (Predict Mode Only)
    # If these don't exist, we must switch process to 'train'
    required_files = ["binning_train.pkl", "logit_model.pkl", "calibrated_model.pkl"]
    missing_files = [f for f in required_files if not (MODEL_DIR / f).exists()]

    mode = 'predict'
    if missing_files:
        print(f"⚠️ Missing artifacts: {missing_files}")
        print("Switching to 'train' mode to generate required models...")
        mode = 'train'
        
        # Training requires a target. If your sample doesn't have it, we add dummy data.
        df_check = pd.read_csv(DATA_IN)
        if 'loan_status' not in df_check.columns:
            print("Adding dummy 'loan_status' for training compatibility...")
            df_check['loan_status'] = np.random.randint(0, 2, df_check.shape[0])
            df_check.to_csv(DATA_IN, index=False)

    # 3. EXECUTION
    try:
        print(f"Running Pricing Pipeline in [{mode.upper()}] mode...")
        
        # Calling the function with absolute path string for Docker/Root compatibility
        run_pricing_pipeline(
            data_path=str(DATA_IN.resolve()), 
            process=mode
        )
        
        print(f"✅ Price Engine {mode} complete.")
        
        # Verify Output
        output_path = BASE_DIR / "data" / "models" / "price_engine" / "initial_risk_report.csv"
        if output_path.exists():
            print(f"✅ Success! Risk Report generated at: {output_path}")
            print(pd.read_csv(output_path).head())
        else:
            print("⚠️ Warning: Pipeline finished but output report was not found.")

    except Exception as e:
        print(f"❌ PIPELINE FAILURE: {e}")

if __name__ == "__main__":
    main()