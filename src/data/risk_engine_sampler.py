import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/risk_engine_sampler.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()

def generate_risk_samples():
    """
    Orchestrates the merging of approved loans with additional features 
    from the test set to prepare the final Risk Engine input.
    """
    # 2. Setup Absolute Paths anchored to ROOT for Docker volume consistency
    filtered_path = ROOT / "data" / "generated" / "sample_filtered.csv"
    test_path = ROOT / "data" / "cleaned" / "splitter" / "test.csv"
    output_path = ROOT / "data" / "generated" / "risk_engine_sample_generated.csv"

    

    try:
        # 3. Load the Data
        if not filtered_path.exists():
            print(f"❌ Error: Filtered data not found at {filtered_path}")
            return
        
        df_filtered = pd.read_csv(filtered_path)
        target_size = len(df_filtered)
        
        if not test_path.exists():
            print(f"❌ Error: Test data not found at {test_path}")
            return
            
        df_test = pd.read_csv(test_path)

        # 4. Random Sampling
        # We sample features from the master test set to match our approved cohort size.
        # replace=True ensures we don't crash if the approved batch exceeds test set size.
        df_sampled = df_test.sample(n=target_size, replace=True, random_state=42).reset_index(drop=True)

        # 5. Horizontal Merging (Feature Alignment)
        # Identify columns in the test sample that are NOT already in our filtered cohort.
        unique_cols = [col for col in df_sampled.columns if col not in df_filtered.columns]
        
        # Concatenate horizontally on axis=1. Resetting index is vital to prevent NaNs.
        result_df = pd.concat([df_filtered.reset_index(drop=True), df_sampled[unique_cols]], axis=1)

        # 6. Pre-Processing Cleanup
        # Dropping 'int_rate' to avoid data leakage or redundancy before the pricing engine run.
        if 'int_rate' in result_df.columns:
            result_df = result_df.drop(columns=['int_rate'])
            print("💡 Successfully dropped 'int_rate' column for clean engine input.")

        # 7. Output to CSV
        # Ensure the directory exists inside the container before writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        print(f"✅ Success: Generated {target_size} rows with {len(result_df.columns)} features.")
        print(f"🚀 Saved to: {output_path}")

    except Exception as e:
        print(f"❌ An error occurred during sampling: {e}")

if __name__ == "__main__":
    generate_risk_samples()