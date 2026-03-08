import pandas as pd
import json
from pathlib import Path

def generate_bank_action_reports(df_action, simulation_csv_path):
    """
    Calculates ECL with Stressed Scenarios. Merges risk tags with financial data.
    """ 
    try:
        DATA_DIR = Path(__file__).resolve().parent / "../../../data/models/markov_chains/"
        JSON_DIR = Path(__file__).resolve().parent / "../../../json_files/markov_chains/"
        DATA_DIR.mkdir(parents=True, exist_ok=True); JSON_DIR.mkdir(parents=True, exist_ok=True)
        
        sim_df = pd.read_csv(simulation_csv_path)
        
        # 1. Base LGD Calculation
        df_action['base_lgd'] = (1 - (df_action['recoveries'] / df_action['funded_amnt'])).fillna(0.45)
        df_action['base_lgd'] = df_action['base_lgd'].clip(0, 1)

        # 2. Merge Data (FIXED: Ensure status_tag and cumulative_pd are preserved)
        # We merge all of sim_df with the specific financial columns from df_action
        merged_df = pd.merge(
            sim_df, 
            df_action[['id', 'provision_needed', 'base_lgd']], 
            on='id', 
            how='inner'
        )
        
        if 'status_tag' not in merged_df.columns:
            raise KeyError("Critical error: 'status_tag' not found after merging simulation data.")

        # Scenario Logic
        merged_df['lgd_lower'] = (merged_df['base_lgd'] * 0.9).clip(0, 1)
        merged_df['lgd_upper'] = (merged_df['base_lgd'] * 1.2).clip(0, 1)

        results = {"warning": {"base": 0.0, "lower": 0.0, "upper": 0.0},
                   "attention": {"base": 0.0, "lower": 0.0, "upper": 0.0}}
        late_actions, detailed_rows = [], []

        # 3. ECL and Action Logic
        for _, row in merged_df.iterrows():
            pd_val, balance, tag = row['cumulative_pd'], row['provision_needed'], row['status_tag']
            
            ecl_base = round(balance * pd_val * row['base_lgd'], 2)
            ecl_lower = round(balance * pd_val * row['lgd_lower'], 2)
            ecl_upper = round(balance * pd_val * row['lgd_upper'], 2)

            if tag in results:
                results[tag]["base"] += ecl_base
                results[tag]["lower"] += ecl_lower
                results[tag]["upper"] += ecl_upper

            # Define Action Strings
            action_str = "monitor"
            if tag == 'late':
                action_str = "direct outreach" if pd_val > 0.15 else "automated reminder"
                late_actions.append({"id": row['id'], "action": action_str})
            elif tag == 'attention':
                action_str = "escalate to collections" if pd_val > 0.40 else "soft call"
            elif tag == 'warning':
                action_str = "legal recovery initiation" if pd_val > 0.50 else "hard collection"

            row_dict = row.to_dict()
            row_dict.update({'action': action_str, 'ecl_base': ecl_base, 'ecl_upper': ecl_upper})
            detailed_rows.append(row_dict)

        # 4. Final JSON Assembly
        summary_metrics = {
            "provisioning_report": {
                "warning_level": {
                    "optimistic_lower_bound": round(results["warning"]["lower"], 2),
                    "primary_requirement_base": round(results["warning"]["base"], 2),
                    "stressed_upper_bound": round(results["warning"]["upper"], 2)
                },
                "attention_level": {
                    "optimistic_lower_bound": round(results["attention"]["lower"], 2),
                    "estimated_need_base": round(results["attention"]["base"], 2),
                    "stressed_upper_bound": round(results["attention"]["upper"], 2)
                }
            }
        }

        with open(JSON_DIR / "bank_provisioning_metrics.json", 'w') as f:
            json.dump(summary_metrics, f, indent=4)
        with open(JSON_DIR / "late_borrower_actions.json", 'w') as f:
            json.dump(late_actions, f, indent=4)
        
        pd.DataFrame(detailed_rows).to_csv(DATA_DIR / "detailed_action_report.csv", index=False)
        return summary_metrics

    except Exception as e:
        print(f"Error in action logic: {e}")
        return None