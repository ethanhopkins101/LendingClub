import pandas as pd
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/opt_metrics.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()

def calculate_optimal_metrics():
    """
    Fetches the result of the PuLP optimization from optimal.csv,
    calculates post-optimization KPIs, and saves them to data/generated/opt_metrics.csv.
    """
    # 2. Setup Absolute Paths anchored to ROOT for Docker volume consistency
    input_path = ROOT / "data" / "generated" / "optimal.csv"
    output_path = ROOT / "data" / "generated" / "opt_metrics.csv"

    

    try:
        # 3. Check and Load Data
        if not input_path.exists():
            print(f"❌ Error: Optimal portfolio file not found at {input_path}")
            return

        df = pd.read_csv(input_path)

        # Handle empty optimization result (e.g., if constraints were too strict)
        if df.empty:
            print("⚠️ Warning: optimal.csv is empty. No loans met the constraints.")
            return

        # 4. Aggregate Absolute Totals
        total_ead = df['EAD'].sum()
        total_el = df['Expected_Loss'].sum()
        total_rev = df['Expected_Revenue'].sum()
        total_profit = df['Expected_Profit'].sum()

        # 5. Industry Standard KPIs
        # Portfolio Loss Rate (EL / EAD): Measures overall credit quality
        portfolio_loss_rate = total_el / total_ead if total_ead != 0 else 0
        
        # Risk-Adjusted Margin (Profit / Revenue): Measures efficiency of earnings
        net_profit_margin = total_profit / total_rev if total_rev != 0 else 0
        
        # Weighted averages for quality assessment
        avg_pd = df['probability_of_default'].mean()
        avg_credit_score = df['credit_score'].mean()
        
        # Portfolio Yield (Revenue / EAD): Gross return before losses
        portfolio_yield = total_rev / total_ead if total_ead != 0 else 0

        # 6. Create Metrics DataFrame
        metrics_data = {
            "Metric": [
                "Total Exposure at Default (EAD)",
                "Total Expected Loss (EL)",
                "Total Expected Revenue",
                "Total Expected Profit",
                "Portfolio Loss Rate (%)",
                "Net Profit Margin (%)",
                "Average Probability of Default",
                "Average Credit Score",
                "Portfolio Yield (%)"
            ],
            "Value": [
                round(total_ead, 2),
                round(total_el, 2),
                round(total_rev, 2),
                round(total_profit, 2),
                round(portfolio_loss_rate * 100, 2),
                round(net_profit_margin * 100, 2),
                round(avg_pd, 4),
                round(avg_credit_score, 0),
                round(portfolio_yield * 100, 2)
            ]
        }

        metrics_df = pd.DataFrame(metrics_data)

        # 7. Save Output
        # Ensure the directory exists inside the container before writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)

        print(f"✅ Success: Optimized metrics saved to {output_path}")
        return metrics_df

    except Exception as e:
        print(f"❌ Error calculating optimized metrics: {e}")

if __name__ == "__main__":
    calculate_optimal_metrics()