import pandas as pd
import sys
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/markovian_sampler.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()

def run_stratified_sample(sample_size=30):
    """
    Performs stratified sampling based on loan_status to ensure 
    transition matrices represent all states correctly.
    """
    # 2. Setup Absolute Paths anchored to ROOT
    INPUT_PATH = ROOT / "data" / "cleaned" / "markovian" / "markov_isolater.csv"
    OUTPUT_DIR = ROOT / "data" / "generated"
    OUTPUT_FILE = OUTPUT_DIR / "markovian_sample.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    

    try:
        # 3. Load the Isolated Markovian Data
        if not INPUT_PATH.exists():
            print(f"❌ Error: Input file not found at {INPUT_PATH}")
            return

        df = pd.read_csv(INPUT_PATH)
        
        # 4. Stratified Sampling Logic
        # Grouping by 'loan_status' ensures that minority classes (like 'Default') 
        # are proportionally represented in the sample.
        total_rows = len(df)
        if sample_size > total_rows:
            print(f"⚠️ Warning: Requested {sample_size} but only {total_rows} available.")
            sample_size = total_rows

        fraction = sample_size / total_rows

        # Apply random sampling within each stratum
        sampled_df = df.groupby("loan_status", group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=42)
        )

        # Enforce exact sample size requested
        sampled_df = sampled_df.head(sample_size)

        # 5. Column Reordering Logic
        # Moving identifiers and target variables to the front for UI readability
        priority_cols = ['id', 'loan_status', 'loan_amnt', 'funded_amnt']
        other_cols = [c for c in sampled_df.columns if c not in priority_cols]
        final_column_order = [c for c in priority_cols if c in sampled_df.columns] + other_cols
        sampled_df = sampled_df[final_column_order]
        
        # 6. Save Results
        sampled_df.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Stratified sample of {len(sampled_df)} rows saved to: {OUTPUT_FILE}")
        print(f"🔝 Columns reordered. Primary identifiers moved to front.")

    except Exception as e:
        print(f"❌ Backend Error: {e}")

if __name__ == "__main__":
    # Support for CLI arguments when called via subprocess in backend.py
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
            run_stratified_sample(size)
        except ValueError:
            print("⚠️ Invalid input. Running with default size (30).")
            run_stratified_sample()
    else:
        run_stratified_sample()