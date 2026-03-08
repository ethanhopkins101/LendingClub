import pandas as pd
import numpy as np
from pathlib import Path

def generate_final_pricing(csv_path):
    """
    Applies market-adjusted risk premiums and US interest rate caps.
    Saves the final pricing report to the specified directory.
    """
    try:
        # 1. Load Initial Risk Report (Ensure path is handled correctly)
        abs_csv_path = Path(csv_path).resolve()
        df = pd.read_csv(abs_csv_path)
        
        # 2. Market-Calibrated Risk Premium
        # Factor of 45: Translates Probability of Default into a percentage spread
        df['risk_premium_rate'] = (df['pd'] * 45).round(2)

        # 3. Final Interest Rate Calculation
        # Final Rate = Purpose-Based Base Rate + Individual Risk Premium
        df['int_rate'] = (df['base_interest_rate_pct'] + df['risk_premium_rate']).round(2)

        # 4. Regulatory Compliance: USA Usury Cap (35.99%)
        # Clips any rate exceeding the legal limit to 35.99%
        df['int_rate'] = df['int_rate'].clip(upper=35.99)

        # 5. Filter for Final Output
        final_cols = ['id', 'base_interest_rate_pct', 'risk_premium_rate', 'int_rate']
        final_pricing_df = df[final_cols].copy()

        # 6. Save Final CSV to Absolute Path
        SAVE_DIR = Path(__file__).resolve().parent / "../../../data/models/price_engine/"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        output_file = SAVE_DIR / "final_pricing_report.csv"
        final_pricing_df.to_csv(output_file, index=False)
        
        print(f"Final pricing report generated successfully at: {output_file}")
        return final_pricing_df

    except Exception as e:
        print(f"Error in generating final pricing: {e}")
        return None