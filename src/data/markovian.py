import pandas as pd
from pathlib import Path

def prepare_markovian_ongoing_data():
    """
    Filters for ongoing loans and returns a sample of 50,000 records.
    """
    try:
        # 1. Setup Paths (Fixed to reach root data folder)
        input_path = Path(__file__).resolve().parent / "../../data/clean/accepted_cleaned_full.csv"
        output_dir = Path(__file__).resolve().parent / "../../data/cleaned/markovian/"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"Master cleaned data not found at: {input_path}")

        # 2. Load Master Data
        df = pd.read_csv(input_path)

        # 3. Filter for Ongoing States
        status_map = {
            "Current": 0,
            "In Grace Period": 1,
            "Late (16-30 days)": 2,
            "Late (31-120 days)": 3
        }
        ongoing_df = df[df['loan_status'].isin(status_map.keys())].copy()

        # 4. Sampling (50k limit)
        # Use random_state=42 for reproducibility across runs
        if len(ongoing_df) > 50000:
            ongoing_df = ongoing_df.sample(n=50000, random_state=42).reset_index(drop=True)
            print(f"Sampled 50,000 records from {len(df)} initial rows.")
        else:
            print(f"Retained all {len(ongoing_df)} available ongoing records.")

        # 5. Save to Disk
        output_file = output_dir / "ongoing_loans.csv"
        ongoing_df.to_csv(output_file, index=False)

        print(f"Success! Data saved to: {output_file}")
        return ongoing_df

    except Exception as e:
        print(f"Error in markovian data preparation: {e}")
        return None

if __name__ == "__main__":
    prepare_markovian_ongoing_data()