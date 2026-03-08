# import os
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # --- LOCAL IMPORTS ---
# from data_gathering import load_initial_review_data
# from features import (
#     clean_initial_review_data, 
#     split_initial_review_data, 
#     fit_transform_encoders, 
#     apply_encoder_to_df
# )
# from train import (
#     optimize_initial_review_xgb, 
#     train_and_save_final_model, 
#     train_and_save_calibrated_model
# )
# from predictions import (
#     generate_initial_review_predictions, 
#     generate_strategy_analysis
# )

# def run_full_pipe(data_path, threshold=0.5):
#     """
#     Orchestrator for the Initial Review Model.
#     """
#     current_dir = Path(__file__).resolve().parent
#     model_dir = (current_dir / "../../../models/initial_review/").resolve()
#     json_dir = (current_dir / "../../../json_files/initial_review/").resolve()
    
#     model_dir.mkdir(parents=True, exist_ok=True)
#     json_dir.mkdir(parents=True, exist_ok=True)
    
#     xgb_path = model_dir / "xgb_initial_review.pkl"
#     calibrated_path = model_dir / "calibrated_xgb_initial_review.pkl"
#     params_path = json_dir / "best_params.json"
#     encoder_path = model_dir / "ordinal_encoder.pkl"

#     models_exist = calibrated_path.exists() and encoder_path.exists()

#     absolute_data_path = (current_dir / data_path).resolve()
#     df_raw = load_initial_review_data(absolute_data_path)
#     df_clean = clean_initial_review_data(df_raw)

#     if not models_exist:
#         print("--- [TRAIN PIPE] Starting Optimization ---")
        
#         X_train, X_test, y_train, y_test = split_initial_review_data(df_clean)
        
#         # Fit encoders on training data
#         X_train_enc, X_test_enc = fit_transform_encoders(X_train, X_test)
        
#         # Optimize
#         optimize_initial_review_xgb(X_train_enc, X_test_enc, y_train, y_test)
        
#         # Final Training
#         df_encoded = apply_encoder_to_df(df_clean.copy(), encoder_path)
#         train_and_save_final_model(df_encoded, params_path)
#         train_and_save_calibrated_model(df_encoded, xgb_path)
        
#         # Post-Train Strategy CSV generation
#         print("--- [POST-TRAIN] Generating Strategy CSV ---")
#         test_df_labeled = X_test_enc.copy()
#         test_df_labeled['loan_status'] = y_test.values # Ensure alignment
#         generate_strategy_analysis(test_df_labeled, calibrated_path)
        
#         print("--- [TRAIN PIPE] Completed Successfully ---")

#     else:
#         print("--- [PREDICT PIPE] Artifacts detected. ---")
#         df_encoded = apply_encoder_to_df(df_clean.copy(), encoder_path)
        
#         features_only = df_encoded.drop(columns=['loan_status'], errors='ignore')
#         generate_initial_review_predictions(
#             df=features_only, 
#             model_pkl_path=calibrated_path, 
#             threshold=threshold
#         )
        
#         if 'loan_status' in df_encoded.columns:
#             generate_strategy_analysis(df_encoded, calibrated_path)
#         else:
#             print("Notice: Strategy CSV skipped (Data is unlabeled).")

#         print("--- [PREDICT PIPE] Completed Successfully ---")



# if __name__ == "__main__":
#     # Path to training data to ensure Strategy CSV is triggered
#     DATA_IN = "../../../data/generated/sample_data.csv"
#     run_full_pipe(DATA_IN, threshold=0.45)


import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve PROJECT_ROOT relative to this file's location
# This ensures it works regardless of where the terminal/Docker starts
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (BASE_DIR / "../../../").resolve()

# Ensure the project root is in the sys.path so local imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Adjust local imports to be absolute from the project root
from src.models.initial_review.data_gathering import load_initial_review_data
from src.models.initial_review.features import (
    clean_initial_review_data, 
    split_initial_review_data, 
    fit_transform_encoders, 
    apply_encoder_to_df
)
from src.models.initial_review.train import (
    optimize_initial_review_xgb, 
    train_and_save_final_model, 
    train_and_save_calibrated_model
)
from src.models.initial_review.predictions import (
    generate_initial_review_predictions, 
    generate_strategy_analysis
)

def run_full_pipe(data_path, threshold=0.5, process='predict'):
    """
    Orchestrator for Initial Review.
    :param data_path: Path to the CSV.
    :param threshold: Probability threshold for approval.
    :param process: 'train' to build models, 'predict' to use existing models.
    """
    
    # 2. DEFINE ARTIFACT LOCATIONS - Using absolute PROJECT_ROOT
    model_dir = PROJECT_ROOT / "models" / "initial_review"
    json_dir = PROJECT_ROOT / "json_files" / "initial_review"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    
    xgb_path = model_dir / "xgb_initial_review.pkl"
    calibrated_path = model_dir / "calibrated_xgb_initial_review.pkl"
    params_path = json_dir / "best_params.json"
    encoder_path = model_dir / "ordinal_encoder.pkl"

    # 3. RESOLVE INPUT PATH
    target_path = Path(data_path)
    # Ensure data_path is treated as relative to PROJECT_ROOT if not absolute
    absolute_data_path = target_path if target_path.is_absolute() else (PROJECT_ROOT / data_path).resolve()

    if not absolute_data_path.exists():
        raise FileNotFoundError(f"Input data not found at: {absolute_data_path}")

    # 4. LOAD & CLEAN
    df_raw = load_initial_review_data(absolute_data_path)
    df_clean = clean_initial_review_data(df_raw)

    # --- 5. EXECUTION LOGIC BASED ON PROCESS PARAMETER ---
    
    if process == 'train':
        print(f"--- [TRAIN MODE] Initializing Training Pipeline ---")
        if 'loan_status' not in df_clean.columns:
            raise ValueError("Training requires 'loan_status' column in the input data.")

        X_train, X_test, y_train, y_test = split_initial_review_data(df_clean)
        X_train_enc, X_test_enc = fit_transform_encoders(X_train, X_test)
        
        optimize_initial_review_xgb(X_train_enc, X_test_enc, y_train, y_test)
        
        df_encoded = apply_encoder_to_df(df_clean.copy(), encoder_path)
        train_and_save_final_model(df_encoded, params_path)
        train_and_save_calibrated_model(df_encoded, xgb_path)
        
        # Strategy report for validation
        test_df_labeled = X_test_enc.copy()
        test_df_labeled['loan_status'] = y_test.values
        generate_strategy_analysis(test_df_labeled, calibrated_path)
        print("--- [TRAIN MODE] Completed Successfully ---")

    elif process == 'predict':
        print(f"--- [PREDICT MODE] Using Artifacts from {model_dir} ---")
        
        # Check if we actually have the models to predict with
        if not (calibrated_path.exists() and encoder_path.exists()):
            raise FileNotFoundError(f"Missing model artifacts in {model_dir}. You must run 'train' process first.")

        df_encoded = apply_encoder_to_df(df_clean.copy(), encoder_path)
        
        # Drop target if it exists to simulate real-world prediction
        features_only = df_encoded.drop(columns=['loan_status'], errors='ignore')
        
        generate_initial_review_predictions(
            df=features_only, 
            model_pkl_path=calibrated_path, 
            threshold=threshold
        )
        
        # If ground truth is available in the data, we can also update strategy analysis
        if 'loan_status' in df_encoded.columns:
            generate_strategy_analysis(df_encoded, calibrated_path)
            
        print("--- [PREDICT MODE] Completed Successfully ---")
    
    else:
        raise ValueError("Invalid process type. Choose 'train' or 'predict'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/generated/sample_data.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default="predict", choices=['train', 'predict'])
    args = parser.parse_args()

    run_full_pipe(data_path=args.data, threshold=args.threshold, process=args.mode)