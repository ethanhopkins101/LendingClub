# import os
# from pathlib import Path
# import pandas as pd

# # Internal Imports
# from data_gathering import load_and_route_data
# from features import (handle_missing_values, engineer_features, 
#                       calculate_base_rates, prepare_and_bin_data, 
#                       split_training_data, sync_dataframes_by_id)
# from train import (optimize_and_save_params, train_final_logit, 
#                    calibrate_and_save)
# from initial_pricing import generate_initial_risk_report
# from final_pricing import generate_final_pricing

# # Absolute Paths
# BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = (BASE_DIR / "../../../").resolve()

# MODEL_DIR = PROJECT_ROOT / "models/price_engine"
# LGD_MODEL_PATH = PROJECT_ROOT / "models/probability_of_default/final_lgd_model.pkl"
# JSON_PARAM_PATH = PROJECT_ROOT / "json_files/price_engine/best_params.json"

# # Specific Binning Artifacts
# BIN_TRAIN_PKL = MODEL_DIR / "binning_train.pkl"
# BIN_PREDICT_PKL = MODEL_DIR / "binning_predict.pkl"

# LOGIT_PKL_PATH = MODEL_DIR / "logit_model.pkl"
# CAL_PKL_PATH = MODEL_DIR / "calibrated_model.pkl"
# RISK_REPORT_CSV = PROJECT_ROOT / "data/models/price_engine/initial_risk_report.csv"

# def run_pricing_pipeline(data_path, interest_json_path=None):
#     abs_data_path = (PROJECT_ROOT / data_path).resolve()
    
#     if not abs_data_path.exists():
#         raise FileNotFoundError(f"Input data not found at: {abs_data_path}")

#     # Determine mode based on existence of TRAINED models
#     is_prediction_ready = BIN_TRAIN_PKL.exists() and CAL_PKL_PATH.exists()

#     if not is_prediction_ready:
#         print("--- Mode: TRAINING ---")
#         df = load_and_route_data(abs_data_path)
        
#         df_cleaned = handle_missing_values(df)
#         df_engineered = engineer_features(df_cleaned)
#         df_base = calculate_base_rates(df_engineered, interest_json_path)

#         # 1. Prepare and Bin (Saves binning_train.pkl)
#         df_ids, df_woe, y = prepare_and_bin_data(df_base, mode='train')
#         X_train, X_test, y_train, y_test = split_training_data(df_woe, y)

#         # 2. Train and Calibrate
#         optimize_and_save_params(X_train, X_test, y_train, y_test)
#         train_final_logit(df_woe.join(y), str(JSON_PARAM_PATH), str(BIN_TRAIN_PKL))
#         calibrate_and_save(df_woe.join(y), str(BIN_TRAIN_PKL), str(LOGIT_PKL_PATH))
        
#         print("Training Pipeline Complete.")

#     else:
#         print("--- Mode: PREDICTION ---")
#         # 1. Data Gathering (Expects PD, LGD)
#         pd_data, lgd_data = load_and_route_data(abs_data_path)

#         # 2. Features Logic on PD data
#         pd_cleaned = handle_missing_values(pd_data)
#         pd_engineered = engineer_features(pd_cleaned)

#         # 3. Sync and Base Rates
#         pd_synced, lgd_synced = sync_dataframes_by_id(pd_engineered, lgd_data)
#         pd_base = calculate_base_rates(pd_synced, interest_json_path)
        
#         # 4. PREPARE AND BIN PREDICTION DATA
#         # This saves binning_predict.pkl specifically for this batch
#         _, pd_cleaned_base_binned, _ = prepare_and_bin_data(pd_base, mode='predict')

#         # 5. INITIAL PRICING
#         # Pass BIN_PREDICT_PKL so the model uses the bins just created for this data
#         generate_initial_risk_report(
#             pd_base,        # Cleaned PD Data with Base Rate
#             lgd_synced,     # Synced LGD Data
#             str(CAL_PKL_PATH), 
#             str(BIN_PREDICT_PKL), # USING THE PREDICT BINNING MODEL
#             str(LGD_MODEL_PATH)
#         )

#         # 6. FINAL PRICING
#         generate_final_pricing(str(RISK_REPORT_CSV))
#         print("Inference Pipeline Complete.")

# if __name__ == "__main__":
#     # Pointing to the test data for prediction mode
#     TARGET_DATA = "data/generated/risk_engine_sample_generated.csv"
#     run_pricing_pipeline(TARGET_DATA)

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve PROJECT_ROOT relative to this file to handle Docker containers properly
BASE_DIR = Path(__file__).resolve().parent
# Looking up three levels to find the project root: src/models/price_engine/execution.py
PROJECT_ROOT = (BASE_DIR / "../../../").resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 2. ABSOLUTE IMPORTS FROM ROOT ---
from src.models.price_engine.data_gathering import load_and_route_data
from src.models.price_engine.features import (
    handle_missing_values, engineer_features, 
    calculate_base_rates, prepare_and_bin_data, 
    split_training_data, sync_dataframes_by_id
)
from src.models.price_engine.train import (
    optimize_and_save_params, train_final_logit, 
    calibrate_and_save
)
from src.models.price_engine.initial_pricing import generate_initial_risk_report
from src.models.price_engine.final_pricing import generate_final_pricing

# --- 3. DOCKER-READY PATHS ---
# Using PROJECT_ROOT ensures these paths work within Docker Volumes
MODEL_DIR = PROJECT_ROOT / "models" / "price_engine"
LGD_MODEL_PATH = PROJECT_ROOT / "models" / "probability_of_default" / "final_lgd_model.pkl"
JSON_PARAM_PATH = PROJECT_ROOT / "json_files" / "price_engine" / "best_params.json"

BIN_TRAIN_PKL = MODEL_DIR / "binning_train.pkl"
BIN_PREDICT_PKL = MODEL_DIR / "binning_predict.pkl"
LOGIT_PKL_PATH = MODEL_DIR / "logit_model.pkl"
CAL_PKL_PATH = MODEL_DIR / "calibrated_model.pkl"

RISK_REPORT_DIR = PROJECT_ROOT / "data" / "models" / "price_engine"
RISK_REPORT_CSV = RISK_REPORT_DIR / "initial_risk_report.csv"

def run_pricing_pipeline(data_path, interest_json_path=None, process='predict'):
    """Orchestrator for the Price Engine."""
    # Ensure output directories exist inside the container
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RISK_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    target_path = Path(data_path)
    # Resolve relative paths against PROJECT_ROOT for Docker volume consistency
    abs_data_path = target_path if target_path.is_absolute() else (PROJECT_ROOT / data_path).resolve()
    
    if not abs_data_path.exists():
        raise FileNotFoundError(f"Input data not found at: {abs_data_path}")

    # --- MODE: TRAINING ---
    if process == 'train':
        print("--- [PRICE ENGINE] Mode: TRAINING ---")
        df = load_and_route_data(abs_data_path)
        df_cleaned = handle_missing_values(df)
        df_engineered = engineer_features(df_cleaned)
        df_base = calculate_base_rates(df_engineered, interest_json_path)

        df_ids, df_woe, y = prepare_and_bin_data(df_base, mode='train')
        X_train, X_test, y_train, y_test = split_training_data(df_woe, y)

        optimize_and_save_params(X_train, X_test, y_train, y_test)
        train_final_logit(df_woe.join(y), str(JSON_PARAM_PATH), str(BIN_TRAIN_PKL))
        calibrate_and_save(df_woe.join(y), str(BIN_TRAIN_PKL), str(LOGIT_PKL_PATH))
        print("--- [PRICE ENGINE] Training Pipeline Complete ---")

    # --- MODE: PREDICTION ---
    elif process == 'predict':
        print("--- [PRICE ENGINE] Mode: PREDICTION ---")
        
        if not (BIN_TRAIN_PKL.exists() and CAL_PKL_PATH.exists()):
            raise FileNotFoundError(f"Missing price engine artifacts in {MODEL_DIR}. Run 'train' first.")
        
        pd_data, lgd_data = load_and_route_data(abs_data_path)
        pd_cleaned = handle_missing_values(pd_data)
        pd_engineered = engineer_features(pd_cleaned)
        pd_synced, lgd_synced = sync_dataframes_by_id(pd_engineered, lgd_data)
        
        if pd_synced.empty:
            raise ValueError("No matching IDs found between PD and LGD data.")
            
        pd_base = calculate_base_rates(pd_synced, interest_json_path)
        _, pd_cleaned_base_binned, _ = prepare_and_bin_data(pd_base, mode='predict')

        if isinstance(pd_base, pd.Series):
            pd_base = pd_base.to_frame()
            
        generate_initial_risk_report(
            pd_base, 
            lgd_synced, 
            str(CAL_PKL_PATH), 
            str(BIN_PREDICT_PKL), 
            str(LGD_MODEL_PATH)
        )

        generate_final_pricing(str(RISK_REPORT_CSV))
        print("--- [PRICE ENGINE] Inference Pipeline Complete ---")

# --- 4. EXECUTION GATEKEEPER ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--mode", type=str, default="predict", choices=["train", "predict"])
    args = parser.parse_args()

    run_pricing_pipeline(data_path=args.data, process=args.mode)