import pandas as pd
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/filter_price_engine.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()

def filter_approved_loans():
    """
    Filters the initial sample data to only include loans approved by the 
    Initial Review model (initial_prediction == 0).
    """
    # 2. Setup Absolute Paths anchored to ROOT for Docker volume consistency
    results_path = ROOT / "data" / "models" / "initial_review" / "initial_review_results.csv"
    source_data_path = ROOT / "data" / "generated" / "sample_data.csv"
    output_path = ROOT / "data" / "generated" / "sample_filtered.csv"

    

    try:
        # 3. Load the results and identify approved indices
        if not results_path.exists():
            print(f"❌ Error: Predictions not found at {results_path}")
            return
            
        results_df = pd.read_csv(results_path)
        
        # Identify Approved loans (where initial_prediction == 0)
        # We assume the first column contains the index/ID alignment
        approved_mask = results_df['initial_prediction'] == 0
        approved_indices = results_df.loc[approved_mask, results_df.columns[0]].tolist()
        
        print(f"✅ Found {len(approved_indices)} approved loans.")

        # 4. Load the source data and filter by those indices
        if not source_data_path.exists():
            print(f"❌ Error: Source data not found at {source_data_path}")
            return

        source_df = pd.read_csv(source_data_path)
        
        # Filter the source dataframe using the identified indices
        filtered_df = source_df.iloc[approved_indices].copy()

        # 5. Save the filtered output
        # Ensure the directory exists inside the container before writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_path, index=False)
        
        print(f"🚀 Successfully saved filtered loans to: {output_path}")

    except Exception as e:
        print(f"❌ An error occurred during filtering: {e}")

if __name__ == "__main__":
    filter_approved_loans()