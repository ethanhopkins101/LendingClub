import pandas as pd
from pathlib import Path

def split_and_downsample_initial_review():
    """
    Creates a 40k training sample (full schema) and a 5k prediction 
    sample (9-column schema) for the Initial Review pipeline.
    """
    try:
        # 1. Path Setup
        BASE_DIR = Path(__file__).resolve().parent
        INPUT_PATH = (BASE_DIR / "../../data/clean/accepted_cleaned_full.csv").resolve()
        OUTPUT_DIR = (BASE_DIR / "../../data/cleaned/splitter/").resolve()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Reading source data from: {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH, low_memory=False)

        # 2. Generate Training Set (40k rows, all available columns)
        # This will trigger the 'Training' pipe because it has > 9 cols and 'loan_status'
        train_df = df.sample(n=40000, random_state=42)
        train_path = OUTPUT_DIR / "initial_review_train_40k.csv"
        train_df.to_csv(train_path, index=False)
        print(f"Saved training sample (40k rows) to: {train_path}")

        # 3. Generate Prediction Set (5k rows, exactly 9 columns)
        # We ensure no overlap to simulate real "new" borrowers
        remaining_df = df.drop(train_df.index)
        pred_sample = remaining_df.sample(n=5000, random_state=7)

        # The 9 specific columns required for the 'Prediction' pipe logic
        prediction_cols = [
            'loan_amnt', 'issue_d', 'title', 'fico_range_high', 
            'dti', 'zip_code', 'addr_state', 'emp_length', 'policy_code'
        ]

        # Verify columns exist before filtering
        valid_cols = [c for c in prediction_cols if c in pred_sample.columns]
        pred_df = pred_sample[valid_cols].copy()
        
        pred_path = OUTPUT_DIR / "initial_review_predict_5k.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved prediction sample (5k rows, 9 cols) to: {pred_path}")

    except Exception as e:
        print(f"Error during splitting: {e}")

if __name__ == "__main__":
    split_and_downsample_initial_review()