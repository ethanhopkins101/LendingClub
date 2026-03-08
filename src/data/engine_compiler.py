import pandas as pd
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/engine_compiler.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()

def compile_final_pricing():
    """
    Merges pricing reports with generated samples into a final CSV.
    Ensures absolute path resolution for Docker container volumes.
    """
    # 2. Setup Absolute Paths anchored to ROOT
    pricing_report_path = ROOT / "data" / "models" / "price_engine" / "final_pricing_report.csv"
    generated_sample_path = ROOT / "data" / "generated" / "risk_engine_sample_generated.csv"
    output_path = ROOT / "data" / "generated" / "final_pricing.csv"

    

    try:
        # 3. Load the Data
        # Using exists() on absolute paths ensures Docker volume mounts are active
        if not pricing_report_path.exists() or not generated_sample_path.exists():
            print(f"❌ Error: Missing input files.")
            print(f"Expected pricing: {pricing_report_path}")
            print(f"Expected samples: {generated_sample_path}")
            return

        df_pricing = pd.read_csv(pricing_report_path)
        df_samples = pd.read_csv(generated_sample_path)

        # 4. Merge Logic (Based on 'id')
        # Left merge keeps only the records that successfully passed through the pricing engine
        final_df = pd.merge(df_pricing, df_samples, on='id', how='left', suffixes=('', '_drop'))

        # 5. Column Reordering & Cleanup
        # Remove redundant columns created by the merge
        final_df = final_df.loc[:, ~final_df.columns.str.contains('_drop')]

        cols = final_df.columns.tolist()
        
        # Ensure 'id' is the leading column for frontend readability
        if 'id' in cols:
            cols.insert(0, cols.pop(cols.index('id')))
        
        final_df = final_df[cols]

        # 6. Output to CSV
        # Ensure the directory exists inside the container before writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        print(f"✅ Success: Compiled final pricing for {len(final_df)} loans.")
        print(f"Final file saved to: {output_path}")

    except Exception as e:
        print(f"❌ An error occurred during compilation: {e}")

if __name__ == "__main__":
    compile_final_pricing()