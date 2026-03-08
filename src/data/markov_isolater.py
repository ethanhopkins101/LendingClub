import pandas as pd
from pathlib import Path
import os

def isolate_markov_data():
    # 1. Setup Absolute Paths
    # Using parent references to navigate from src/data to the root data folder
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_PATH = (BASE_DIR / "../../data/clean/accepted_cleaned_full.csv").resolve()
    OUTPUT_DIR = (BASE_DIR / "../../data/cleaned/markovian/").resolve()
    OUTPUT_FILE = OUTPUT_DIR / "markov_isolater.csv"

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Loading data from: {INPUT_PATH}")

    try:
        # 2. Load the main dataset
        # low_memory=False prevents DtypeWarnings on large financial datasets
        df = pd.read_csv(INPUT_PATH, low_memory=False)

        # 3. Define the target statuses
        target_statuses = [
            "Current",
            "In Grace Period",
            "Late (16-30 days)",
            "Late (31-120 days)"
        ]

        # 4. Sampling Logic
        sampled_frames = []
        
        for status in target_statuses:
            category_df = df[df['loan_status'] == status]
            count = len(category_df)
            
            if count > 0:
                # Sample 10k or whatever is available if less than 10k
                n_samples = min(count, 10000)
                sampled_frames.append(category_df.sample(n=n_samples, random_state=42))
                print(f"✅ Sampled {n_samples} rows for status: {status}")
            else:
                print(f"⚠️ Warning: No rows found for status: {status}")

        # 5. Combine and Save
        if sampled_frames:
            final_df = pd.concat(sampled_frames, axis=0).reset_index(drop=True)
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"💾 Success! Markovian subset saved to: {OUTPUT_FILE}")
            print(f"📊 Total Rows: {len(final_df)}")
        else:
            print("❌ Error: No data matched the target statuses.")

    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at {INPUT_PATH}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    isolate_markov_data()