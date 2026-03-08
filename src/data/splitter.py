import pandas as pd
from pathlib import Path

def split_and_downsample_data():
    """
    Creates a 40k training sample and a 5k prediction sample for pipeline testing.
    """
    try:
        # 1. Path Setup
        BASE_DIR = Path(__file__).resolve().parent
        INPUT_PATH = (BASE_DIR / "../../data/clean/accepted_cleaned_full.csv").resolve()
        OUTPUT_DIR = (BASE_DIR / "../../data/cleaned/splitter/").resolve()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Reading data from: {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH, low_memory=False)

        # 2. Generate Training Set (40k rows, full columns)
        train_df = df.sample(n=40000, random_state=42)
        train_path = OUTPUT_DIR / "train.csv"
        train_df.to_csv(train_path, index=False)
        print(f"Saved training sample (40k rows) to: {train_path}")

        # 3. Generate Prediction Set (5k rows, restricted columns)
        # Select 5k rows NOT in the training set to ensure zero overlap
        remaining_df = df.drop(train_df.index)
        test_sample = remaining_df.sample(n=5000, random_state=7)

        # Define specific features for the prediction flow (No loan_status)
        prediction_cols = [
            'id', 'loan_amnt', 'term', 'grade', 'emp_title', 'home_ownership',
            'annual_inc', 'verification_status', 'dti', 'earliest_cr_line',
            'inq_last_6mths', 'total_rev_hi_lim', 'acc_open_past_24mths',
            'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_rev_tl_op',
            'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
            'mths_since_recent_bc', 'mths_since_recent_inq', 'num_actv_rev_tl',
            'num_tl_op_past_12m', 'purpose', 'addr_state', 'disbursement_method', 'int_rate'
        ]

        # Filter columns that exist in the dataframe
        valid_cols = [c for c in prediction_cols if c in test_sample.columns]
        test_df = test_sample[valid_cols].copy()
        
        test_path = OUTPUT_DIR / "test.csv"
        test_df.to_csv(test_path, index=False)
        print(f"Saved prediction sample (5k rows) to: {test_path}")

    except Exception as e:
        print(f"Error during splitting: {e}")

if __name__ == "__main__":
    split_and_downsample_data()