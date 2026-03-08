import pandas as pd
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/generate_ir_sample.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()

def generate_sample(sample_size: int = 30):
    """
    Samples data from the cleaned master file for initial review.
    Ensures absolute path resolution for Docker container volumes.
    """
    # 2. Build ABSOLUTE paths anchored to ROOT
    input_path = ROOT / "data" / "cleaned" / "splitter" / "initial_review_predict_5k.csv"
    output_dir = ROOT / "data" / "generated"
    output_file = output_dir / "sample_data.csv"

    # 3. Create the 'generated' folder if missing inside the container
    output_dir.mkdir(parents=True, exist_ok=True)

    

    try:
        # 4. Verification for Docker Volume Mounts
        if not input_path.exists():
            print(f"❌ ERROR: File not found at: {input_path}")
            print("Tip: Check if the 'data' volume is correctly mounted in Docker.")
            return

        # 5. Load, Sample, and Save
        df = pd.read_csv(input_path)
        
        # Take random rows to simulate a new batch for initial review
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        sample_df.to_csv(output_file, index=False)
        print(f"✅ SUCCESS: Saved {len(sample_df)} rows to {output_file}")
        
    except Exception as e:
        print(f"⚠️ PYTHON ERROR: {str(e)}")

if __name__ == "__main__":
    generate_sample(30)