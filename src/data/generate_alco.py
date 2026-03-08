import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/generate_alco.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()

def generate_alco_limits():
    """
    Reads portfolio metrics and generates 3 constrained values:
    RWA Limit, Provision Balance, and Liquid Cash Limit.
    Ensures absolute path resolution for Docker container volumes.
    """
    # 2. Setup Absolute Paths anchored to ROOT
    metrics_path = ROOT / "data" / "generated" / "portfolio_metrics.csv"
    output_path = ROOT / "data" / "generated" / "alco_generated.csv"

    

    try:
        if not metrics_path.exists():
            print(f"❌ Error: Metrics not found at {metrics_path}")
            return

        # 3. Extract Key Metrics
        metrics_df = pd.read_csv(metrics_path)
        metrics_dict = dict(zip(metrics_df['Metric'], metrics_df['Value']))

        ead = metrics_dict.get("Total Exposure at Default (EAD)", 0)
        el = metrics_dict.get("Total Expected Loss (EL)", 0)
        revenue = metrics_dict.get("Total Expected Revenue", 0)

        # 4. Math Formulas for Constraints
        # These limits act as the 'boundary' for the linear programming optimizer.
        
        # RWA Limit: Force the model to drop exposure to meet regulatory capital floors.
        rwa_limit = round(ead * np.random.uniform(0.60, 0.75), 2)
        
        # Provision Balance: The budget for Expected Loss. 
        # Forcing a drop here eliminates low-quality/high-risk borrowers.
        provision_limit = round(el * np.random.uniform(0.65, 0.85), 2)
        
        # Liquid Cash Limit: Minimum revenue requirement for liquidity.
        liquid_cash_limit = round(revenue * np.random.uniform(0.50, 0.70), 2)

        # 5. Create ALCO DataFrame
        alco_data = {
            "RWA_Limit": [rwa_limit],
            "Provision_Limit": [provision_limit],
            "Liquid_Cash_Limit": [liquid_cash_limit]
        }
        
        alco_df = pd.DataFrame(alco_data)

        # 6. Save Output
        # Ensure the directory exists inside the container before writing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        alco_df.to_csv(output_path, index=False)

        print(f"✅ ALCO Constraints Generated:")
        print(f"   - RWA Limit: {rwa_limit} (Targeting ~30% reduction)")
        print(f"   - Provision: {provision_limit}")
        print(f"   - Cash Limit: {liquid_cash_limit}")
        print(f"🚀 Saved to: {output_path}")
        
        return alco_df

    except Exception as e:
        print(f"❌ Error generating ALCO limits: {e}")

if __name__ == "__main__":
    generate_alco_limits()