# import os
# import pickle
# import pandas as pd
# import numpy as np
# from pathlib import Path

# from data_gathering import process_ingestion_pipeline 
# from features import (handle_missing_values, engineer_features, 
#                     sync_model_datasets_train, sync_model_datasets_predict,
#                     prepare_and_bin_data, split_training_data, calculate_realized_lgd)
# from train import (optimize_pd_models, train_optimized_models, calibrate_and_save_models, 
#                    optimize_lgd_model, train_final_lgd_model)
# from predictions import calculate_expected_loss_and_profit

# # Paths
# BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = (BASE_DIR / "../../../models/probability_of_default/").resolve()
# JSON_DIR = (BASE_DIR / "../../../json_files/probability_of_default/").resolve()
# DATA_FILE_PATH = BASE_DIR / ".." / ".." / ".." / "data" / "clean" / "accepted_cleaned_full.csv"

# def run_full_pipeline():
#     # 1. Flow Determination: Check for model artifacts
#     logit_path = MODEL_DIR / "logit_pd_calibrated.pkl"
#     lgd_path = MODEL_DIR / "final_lgd_model.pkl"
#     bin_path = MODEL_DIR / "binning_process.pkl"
    
#     models_exist = logit_path.exists() and lgd_path.exists() and bin_path.exists()

#     # 2. Data Gathering: Uses 5-selector logic internally
#     print("--- Gathering and Routing Data ---")
#     ingestion_results = process_ingestion_pipeline(DATA_FILE_PATH)

#     # --- CASE A: TRAINING FLOW (Requires 3 DFs: PD, LGD-Train, LGD-Ongoing) ---
#     if not models_exist:
#         if ingestion_results is None or len(ingestion_results) != 3:
#             print("Error: Training flow requires 3 dataframes. loan_status missing or data invalid.")
#             return

#         df_pd, df_lgd_raw, df_exog_ongoing = ingestion_results
#         print("--- Running Training Pipeline (ID-Blind) ---")

#         # Preprocessing & Engineering
#         df_pd = handle_missing_values(df_pd)
#         df_pd = engineer_features(df_pd)
        
#         # 3-way sync to ensure IDs match across modeling stages
#         df_pd, df_lgd_raw, df_exog_ongoing = sync_model_datasets_train(df_pd, df_lgd_raw, df_exog_ongoing)

#         # PD Modeling: IDs and Targets are stripped inside train.py
#         df_ids_pd, df_binned_pd, y_pd = prepare_and_bin_data(df_pd)
#         X_tr, X_te, y_tr, y_te = split_training_data(df_binned_pd, y_pd)
        
#         optimize_pd_models(X_tr, X_te, y_tr, y_te)
#         logit_raw, xgb_raw = train_optimized_models(df_binned_pd, y_pd, str(JSON_DIR / "best_pd_params.json"))
#         calibrate_and_save_models(logit_raw, xgb_raw, df_binned_pd, y_pd)

#         # LGD Modeling: Targeted features (purpose, addr_state, disbursement_method, term)
#         df_lgd_realized = calculate_realized_lgd(df_lgd_raw)
#         y_lgd = df_lgd_realized['lgd']
#         X_lgd = df_lgd_realized.drop(columns=['lgd'])
        
#         X_tr_l, X_te_l, y_tr_l, y_te_l = split_training_data(X_lgd, y_lgd)
#         optimize_lgd_model(X_tr_l, X_te_l, y_tr_l, y_te_l)
#         train_final_lgd_model(df_lgd_realized, str(JSON_DIR / "best_lgd_params.json"))
        
#         print("--- Training Pipeline Complete. Artifacts saved to /models/ ---")
#         return

#     # --- CASE B: PREDICTION FLOW (Requires 2 DFs: PD, LGD-Exog) ---
#     else:
#         print("--- Models Found: Running Prediction Pipeline (Sort-Predict-Merge) ---")
        
#         # Unpack based on available data (prefers 2-DF pred flow)
#         if len(ingestion_results) == 2:
#             df_pd, df_exog = ingestion_results
#         else:
#             df_pd, _, df_exog = ingestion_results

#         # Preprocessing & Engineering
#         df_pd = handle_missing_values(df_pd)
#         df_pd = engineer_features(df_pd)

#         # Sync prediction sets
#         df_pd, df_exog = sync_model_datasets_predict(df_pd, df_exog)
        
#         # Load binning for WoE transformation
#         with open(bin_path, 'rb') as f:
#             binning_obj = pickle.load(f)
        
#         # Transform features only (Blind to ID/Status)
#         X_input = df_pd.drop(columns=['id', 'loan_status'], errors='ignore')
#         df_woe = binning_obj.transform(X_input, metric="woe")

#         # 4. Financial Prediction & Reporting
#         # IDs are passed here for final re-attachment to arrays
#         print("--- Generating Profitability Report ---")
#         final_report = calculate_expected_loss_and_profit(
#             df_ids=df_pd[['id']].copy(),
#             df_binned=df_woe,
#             df_unbinned=df_pd, 
#             df_exog=df_exog,     
#             pd_model_path=logit_path,
#             lgd_model_path=lgd_path,
#             binning_process_path=bin_path
#         )

#         processed = len(final_report) if final_report is not None else 0
#         print(f"--- Pipeline Successful. Records processed: {processed} ---")
#         return final_report

# if __name__ == "__main__":
#     run_full_pipeline()


# import os
# import pickle
# import pandas as pd
# import numpy as np
# from pathlib import Path

# from data_gathering import process_ingestion_pipeline 
# from features import (handle_missing_values, engineer_features, 
#                     sync_model_datasets_train, sync_model_datasets_predict,
#                     prepare_and_bin_data, split_training_data, calculate_realized_lgd)
# from train import (optimize_pd_models, train_optimized_models, calibrate_and_save_models, 
#                    optimize_lgd_model, train_final_lgd_model)
# from predictions import calculate_expected_loss_and_profit

# # Static Directory Paths
# BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = (BASE_DIR / "../../../models/probability_of_default/").resolve()
# JSON_DIR = (BASE_DIR / "../../../json_files/probability_of_default/").resolve()

# def run_full_pipeline(data_file_path):
#     """
#     Executes the risk pipeline using the provided data file path.
#     Determines flow (Training vs Prediction) based on existing .pkl artifacts.
#     """
#     # 1. Flow Determination
#     logit_path = MODEL_DIR / "logit_pd_calibrated.pkl"
#     lgd_path = MODEL_DIR / "final_lgd_model.pkl"
#     bin_path = MODEL_DIR / "binning_process.pkl"
    
#     models_exist = logit_path.exists() and lgd_path.exists() and bin_path.exists()

#     # 2. Data Gathering
#     print(f"--- Ingesting Data from: {data_file_path} ---")
#     ingestion_results = process_ingestion_pipeline(data_file_path)

#     # --- CASE A: TRAINING FLOW ---
#     if not models_exist:
#         if ingestion_results is None or len(ingestion_results) != 3:
#             print("Error: Training requires 3 DFs (PD, LGD-Train, LGD-Ongoing). Ensure loan_status is present.")
#             return

#         df_pd, df_lgd_raw, df_exog_ongoing = ingestion_results
#         print("--- Mode: Training ---")

#         df_pd = handle_missing_values(df_pd)
#         df_pd = engineer_features(df_pd)
#         df_pd, df_lgd_raw, df_exog_ongoing = sync_model_datasets_train(df_pd, df_lgd_raw, df_exog_ongoing)

#         # PD Modeling
#         df_ids_pd, df_binned_pd, y_pd = prepare_and_bin_data(df_pd)
#         X_tr, X_te, y_tr, y_te = split_training_data(df_binned_pd, y_pd)
        
#         optimize_pd_models(X_tr, X_te, y_tr, y_te)
#         logit_raw, xgb_raw = train_optimized_models(df_binned_pd, y_pd, str(JSON_DIR / "best_pd_params.json"))
#         calibrate_and_save_models(logit_raw, xgb_raw, df_binned_pd, y_pd)

#         # LGD Modeling
#         df_lgd_realized = calculate_realized_lgd(df_lgd_raw)
#         y_lgd = df_lgd_realized['lgd']
#         X_lgd = df_lgd_realized.drop(columns=['lgd'])
        
#         X_tr_l, X_te_l, y_tr_l, y_te_l = split_training_data(X_lgd, y_lgd)
#         optimize_lgd_model(X_tr_l, X_te_l, y_tr_l, y_te_l)
#         train_final_lgd_model(df_lgd_realized, str(JSON_DIR / "best_lgd_params.json"))
        
#         print("--- Training Complete ---")
#         return

#     # --- CASE B: PREDICTION FLOW ---
#     else:
#         print("--- Mode: Prediction ---")
        
#         if len(ingestion_results) == 2:
#             df_pd, df_exog = ingestion_results
#         else:
#             df_pd, _, df_exog = ingestion_results

#         df_pd = handle_missing_values(df_pd)
#         df_pd = engineer_features(df_pd)
#         df_pd, df_exog = sync_model_datasets_predict(df_pd, df_exog)
        
#         with open(bin_path, 'rb') as f:
#             binning_obj = pickle.load(f)
        
#         X_input = df_pd.drop(columns=['id', 'loan_status'], errors='ignore')
#         df_woe = binning_obj.transform(X_input, metric="woe")

#         final_report = calculate_expected_loss_and_profit(
#             df_ids=df_pd[['id']].copy(),
#             df_binned=df_woe,
#             df_unbinned=df_pd, 
#             df_exog=df_exog,     
#             pd_model_path=logit_path,
#             lgd_model_path=lgd_path,
#             binning_process_path=bin_path
#         )

#         print(f"--- Prediction Complete. Rows: {len(final_report) if final_report is not None else 0} ---")
#         return final_report

# if __name__ == "__main__":


#     # TEST_TRAIN_PATH = BASE_DIR / ".." / ".." / ".." / "data" / "cleaned" / "splitter" / "train.csv"
    
#     # # DEBUG LINES:
#     # print(f"Current Script Location: {BASE_DIR}")
#     # print(f"Looking for data at: {TEST_TRAIN_PATH.resolve()}")
    
#     # run_full_pipeline(TEST_TRAIN_PATH.resolve())

#     # Path to the downsampled test data (no loan_status)
#     TEST_PRED_PATH = BASE_DIR / ".." / ".." / ".." / "data" / "cleaned" / "splitter" / "test.csv"
    
#     # Resolve the absolute path to avoid "File Not Found" errors
#     absolute_path = TEST_PRED_PATH.resolve()
    
#     if not absolute_path.exists():
#         print(f"Error: {absolute_path} not found. Did you run splitter.py first?")
#     else:
#         print(f"Targeting Prediction File: {absolute_path}")
#         # This will now trigger CASE B: PREDICTION FLOW
#         run_full_pipeline(absolute_path)


import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT_DIR relative to this file's location to handle Docker Volumes properly
# Path: src/models/probability_of_default/execution.py -> Project Root is 3 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = (BASE_DIR / "../../../").resolve()

# Ensure the script's directory and project root are in sys.path for local and src imports
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data_gathering import process_ingestion_pipeline 
from features import (handle_missing_values, engineer_features, 
                    sync_model_datasets_train, sync_model_datasets_predict,
                    prepare_and_bin_data, split_training_data, calculate_realized_lgd)
from train import (optimize_pd_models, train_optimized_models, calibrate_and_save_models, 
                    optimize_lgd_model, train_final_lgd_model)
from predictions import calculate_expected_loss_and_profit

# --- 2. DOCKER-READY PATHS ---
MODEL_DIR = ROOT_DIR / "models" / "probability_of_default"
JSON_DIR = ROOT_DIR / "json_files" / "probability_of_default"
OUTPUT_DIR = ROOT_DIR / "data" / "models" / "probability_of_default"

def run_full_pipeline(data_file_path):
    """
    Executes the risk pipeline. 
    Saves the final report to data/models/probability_of_default/final_risk_report.csv
    """
    # Create directories if they don't exist inside the container
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    logit_path = MODEL_DIR / "logit_pd_calibrated.pkl"
    lgd_path = MODEL_DIR / "final_lgd_model.pkl"
    bin_path = MODEL_DIR / "binning_process.pkl"
    
    models_exist = logit_path.exists() and lgd_path.exists() and bin_path.exists()

    # Resolve input path against ROOT_DIR if it is relative
    target_path = Path(data_file_path)
    abs_data_path = target_path if target_path.is_absolute() else (ROOT_DIR / data_file_path).resolve()

    # Data Gathering
    print(f"--- Ingesting Data from: {abs_data_path} ---")
    ingestion_results = process_ingestion_pipeline(str(abs_data_path))

    # --- CASE A: TRAINING FLOW ---
    if not models_exist:
        if ingestion_results is None or len(ingestion_results) != 3:
            print("Error: Training requires 3 DFs. Ensure loan_status is present.")
            return

        df_pd, df_lgd_raw, df_exog_ongoing = ingestion_results
        print("--- Mode: Training ---")

        df_pd = handle_missing_values(df_pd)
        df_pd = engineer_features(df_pd)
        df_pd, df_lgd_raw, df_exog_ongoing = sync_model_datasets_train(df_pd, df_lgd_raw, df_exog_ongoing)

        # PD Modeling
        df_ids_pd, df_binned_pd, y_pd = prepare_and_bin_data(df_pd)
        X_tr, X_te, y_tr, y_te = split_training_data(df_binned_pd, y_pd)
        
        optimize_pd_models(X_tr, X_te, y_tr, y_te)
        logit_raw, xgb_raw = train_optimized_models(df_binned_pd, y_pd, str(JSON_DIR / "best_pd_params.json"))
        calibrate_and_save_models(logit_raw, xgb_raw, df_binned_pd, y_pd)

        # LGD Modeling
        df_lgd_realized = calculate_realized_lgd(df_lgd_raw)
        y_lgd = df_lgd_realized['lgd']
        X_lgd = df_lgd_realized.drop(columns=['lgd'])
        
        X_tr_l, X_te_l, y_tr_l, y_te_l = split_training_data(X_lgd, y_lgd)
        optimize_lgd_model(X_tr_l, X_te_l, y_tr_l, y_te_l)
        train_final_lgd_model(df_lgd_realized, str(JSON_DIR / "best_lgd_params.json"))
        
        print("--- Training Complete ---")
        return

    # --- CASE B: PREDICTION FLOW ---
    else:
        print("--- Mode: Prediction ---")
        
        if len(ingestion_results) == 2:
            df_pd, df_exog = ingestion_results
        else:
            df_pd, _, df_exog = ingestion_results

        df_pd = handle_missing_values(df_pd)
        df_pd = engineer_features(df_pd)
        df_pd, df_exog = sync_model_datasets_predict(df_pd, df_exog)
        
        with open(bin_path, 'rb') as f:
            binning_obj = pickle.load(f)
        
        X_input = df_pd.drop(columns=['id', 'loan_status'], errors='ignore')
        df_woe = binning_obj.transform(X_input, metric="woe")

        final_report = calculate_expected_loss_and_profit(
            df_ids=df_pd[['id']].copy(),
            df_binned=df_woe,
            df_unbinned=df_pd, 
            df_exog=df_exog,     
            pd_model_path=str(logit_path),
            lgd_model_path=str(lgd_path),
            binning_process_path=str(bin_path)
        )

        # SAVE THE REPORT TO THE PATH FRONTEND EXPECTS
        output_file = OUTPUT_DIR / "final_risk_report.csv"
        final_report.to_csv(output_file, index=False)

        print(f"--- Prediction Complete. Report saved to: {output_file} ---")
        return final_report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data CSV")
    args = parser.parse_args()
    
    data_path = args.data if args.data else str(ROOT_DIR / "data" / "cleaned" / "splitter" / "test.csv")
    run_full_pipeline(data_path)