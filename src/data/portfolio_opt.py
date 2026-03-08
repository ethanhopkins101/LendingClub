import sys
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from pathlib import Path

# --- 1. DOCKER & ROOT DIR COMPATIBILITY ---
# Resolve ROOT relative to this file's location
# Path: src/data/portfolio_opt.py -> Root is 2 levels up
BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../../").resolve()



def run_portfolio_optimization(input_csv_path, limits_input):
    """
    Solves the optimization problem: Maximize Profit subject to ALCO limits.
    Uses binary integer programming to select the optimal subset of loans.
    """
    # 2. Setup Output Path anchored to ROOT
    output_path = ROOT / "data" / "generated" / "optimal.csv"
    
    try:
        # 3. Load Data (Ensuring input_csv_path is absolute)
        target_input = Path(input_csv_path)
        abs_input_path = target_input if target_input.is_absolute() else (ROOT / input_csv_path).resolve()
        
        if not abs_input_path.exists():
            raise FileNotFoundError(f"Risk report not found at {abs_input_path}")
            
        df = pd.read_csv(abs_input_path)
        
        # 4. Parse Limits (Handle either 3 CLI list values or a CSV path)
        limits = {}
        if isinstance(limits_input, list):
            # Order: RWA, Provision, Liquid Cash
            limits['rwa'] = float(limits_input[0])
            limits['provisioning'] = float(limits_input[1])
            limits['liquidity'] = float(limits_input[2])
        else:
            # Resolve ALCO CSV path relative to ROOT if needed
            target_alco = Path(limits_input)
            abs_alco_path = target_alco if target_alco.is_absolute() else (ROOT / limits_input).resolve()
            
            alco_df = pd.read_csv(abs_alco_path)
            limits['rwa'] = float(alco_df.iloc[0]['RWA_Limit'])
            limits['provisioning'] = float(alco_df.iloc[0]['Provision_Limit'])
            limits['liquidity'] = float(alco_df.iloc[0]['Liquid_Cash_Limit'])

        # 5. Initialize Optimization Problem
        prob = LpProblem("Portfolio_Optimization", LpMaximize)

        # 6. Define Binary Decision Variables
        # x[i] = 1 if loan is kept, 0 if rejected
        loan_indices = df.index.tolist()
        loan_vars = LpVariable.dicts("grant", loan_indices, cat='Binary')

        # 7. Proxy for RWA (Basel III Simplified)
        # Formula: $RWA = EAD \times PD \times 12.5$
        df['RWA'] = df['EAD'] * df['probability_of_default'] * 12.5

        # 8. Objective Function: Maximize Total Expected Profit
        prob += lpSum([df.loc[i, 'Expected_Profit'] * loan_vars[i] for i in loan_indices])

        # 9. Constraints
        # Total Risk Weighted Assets must not exceed regulatory capital buffer
        prob += lpSum([df.loc[i, 'RWA'] * loan_vars[i] for i in loan_indices]) <= limits['rwa']
        
        # Total Expected Loss must be within the provisioning budget
        prob += lpSum([df.loc[i, 'Expected_Loss'] * loan_vars[i] for i in loan_indices]) <= limits['provisioning']
        
        # Total Exposure must stay within liquidity limits
        prob += lpSum([df.loc[i, 'EAD'] * loan_vars[i] for i in loan_indices]) <= limits['liquidity']

        # 10. Solve
        print(f"Solving optimization for {len(df)} loans...")
        prob.solve()

        # 11. Filter and Save Results
        keep_indices = [i for i in loan_indices if value(loan_vars[i]) == 1]
        df_optimal = df.loc[keep_indices].copy()
        
        # Cleanup
        df_optimal = df_optimal.drop(columns=['RWA'])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_optimal.to_csv(output_path, index=False)

        print(f"✅ Optimization Successful.")
        print(f"   Loans Kept: {len(df_optimal)} / {len(df)}")
        print(f"   Optimized Profit: ${df_optimal['Expected_Profit'].sum():,.2f}")
        
        return df_optimal

    except Exception as e:
        print(f"❌ Optimization Error: {e}")
        return None

if __name__ == "__main__":
    # Internal path resolution for CLI usage
    raw_report_path = ROOT / "data" / "models" / "probability_of_default" / "final_risk_report.csv"
    
    if len(sys.argv) == 4:
        # Manual Mode: 3 values passed (rwa, prov, cash)
        run_portfolio_optimization(raw_report_path, sys.argv[1:4])
    elif len(sys.argv) == 2:
        # CSV Mode: Path to ALCO generated constraints passed
        run_portfolio_optimization(raw_report_path, sys.argv[1])
    else:
        print("Usage: python portfolio_opt.py <rwa> <prov> <cash> OR python portfolio_opt.py <alco_csv_path>")