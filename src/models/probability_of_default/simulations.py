import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
import json
from pathlib import Path

def run_portfolio_optimization(df_results, limits):
    """
    Uses Linear Programming (PuLP) to select the optimal subset of loans.
    Objective: Maximize Expected Profit.
    Constraints: Liquidity (EAD), Provisioning (EL), and Regulatory Capital (RWA).
    """
    try:
        # 1. Setup Optimization Problem
        prob = LpProblem("Loan_Selection_Optimization", LpMaximize)

        # 2. Define Variables
        # x[i] is a binary variable: 1 if we grant the loan, 0 otherwise
        loan_ids = df_results['id'].astype(str).tolist()
        loan_vars = LpVariable.dicts("grant", loan_ids, cat='Binary')

        # 3. Add Risk-Weighted Assets (RWA) Proxy
        # Basel III standard: EAD * PD * 12.5
        df_results['RWA'] = df_results['EAD'] * df_results['probability_of_default'] * 12.5

        # 4. Objective Function: Maximize Total Expected Profit
        prob += lpSum([df_results.iloc[i]['Expected_Profit'] * loan_vars[loan_ids[i]] 
                       for i in range(len(df_results))])

        # 5. Constraints (ALCO Limits)
        # Liquidity Constraint: Total EAD <= Liquidity Limit
        prob += lpSum([df_results.iloc[i]['EAD'] * loan_vars[loan_ids[i]] 
                       for i in range(len(df_results))]) <= limits['liquidity']

        # Provisioning Constraint: Total Expected Loss <= Provisioning Limit
        prob += lpSum([df_results.iloc[i]['Expected_Loss'] * loan_vars[loan_ids[i]] 
                       for i in range(len(df_results))]) <= limits['provisioning']

        # Capital Constraint: Total RWA <= RWA Limit
        prob += lpSum([df_results.iloc[i]['RWA'] * loan_vars[loan_ids[i]] 
                       for i in range(len(df_results))]) <= limits['rwa']

        # 6. Solve
        print("Solving Portfolio Optimization Problem...")
        prob.solve()

        # 7. Extract Granted IDs
        granted_ids = [loan_id for loan_id in loan_ids if value(loan_vars[loan_id]) == 1]
        
        # 8. Save Results to JSON
        output_data = {
            "summary": {
                "total_loans_evaluated": len(df_results),
                "total_loans_granted": len(granted_ids),
                "optimized_expected_profit": round(value(prob.objective), 2),
                "status": "Optimal"
            },
            "granted_loan_ids": granted_ids
        }

        BASE_DIR = Path(__file__).resolve().parent
        JSON_DIR = BASE_DIR / "../../../json_files/probability_of_default/"
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        
        save_path = JSON_DIR / "optimal_granted_loans.json"
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"Optimization Complete. {len(granted_ids)} loans approved.")
        print(f"Total Portfolio Profit: ${output_data['summary']['optimized_expected_profit']:,.2f}")
        print(f"IDs saved to: {save_path}")

        return granted_ids

    except Exception as e:
        print(f"Error during simulation: {e}")
        return None