import pandas as pd
import os
import argparse

def generate_ne_table(alphas, base_dir="results"):
    """
    Generates a summary table for Nash Equilibrium analysis with formatted Mean ± Std.
    """
    table_data = []

    # Print header for terminal output
    print(f"{'Alpha':<10} | {'Baseline':<10} | {'Deviated':<10} | {'Regret (Mean ± Std)':<25} | {'Violation %':<12}")
    print("-" * 80)

    for alpha in alphas:
        file_path = os.path.join(base_dir, f"alpha_{alpha}", "best_response_checks.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: Results not found for alpha={alpha}")
            continue

        try:
            df = pd.read_csv(file_path)
            
            # 1. Calculate Metrics
            mean_baseline = df['baseline_reward'].mean()
            mean_deviated = df['best_reward'].mean()
            mean_regret = df['delta'].mean()
            std_regret = df['delta'].std()
            violation_rate = (df['is_ne'] == False).mean() * 100

            # 2. Format the Regret String (e.g., "189.50 ± 45.20")
            regret_str = f"{mean_regret:.2f} ± {std_regret:.2f}"

            # 3. Append to list
            table_data.append({
                "Alpha": alpha,
                "Baseline Reward": mean_baseline,
                "Deviated Reward": mean_deviated,
                "Regret (Mean ± Std)": regret_str,
                "Violation Rate (%)": violation_rate,
                # Keep raw values hidden for sorting if needed, or just for reference
                "_raw_mean_regret": mean_regret 
            })

            # Print row for immediate feedback
            print(f"{alpha:<10} | {mean_baseline:<10.2f} | {mean_deviated:<10.2f} | {regret_str:<25} | {violation_rate:<12.1f}")

        except Exception as e:
            print(f"Error processing alpha={alpha}: {e}")

    # 4. Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    if not df_table.empty:
        # Filter columns for final output
        final_cols = ["Alpha", "Baseline Reward", "Deviated Reward", "Regret (Mean ± Std)", "Violation Rate (%)"]
        df_final = df_table[final_cols]

        # Save to CSV
        output_csv = os.path.join(base_dir, "ne_stability_table.csv")
        df_final.to_csv(output_csv, index=False)
        print(f"\n✅ Table saved to: {output_csv}")
        
        # Print LaTeX code
        print("\n--- LaTeX Code for Paper ---")
        # Note: We use 'MultiIndex' or just column formatting. 
        # Pandas to_latex might escape the ± symbol, so we verify.
        latex_str = df_final.to_latex(
            index=False, 
            float_format="%.2f",
            caption="Nash Equilibrium Stability Analysis. Regret is shown as Mean $\\pm$ Std.",
            label="tab:ne_stability",
            column_format="lcccc" # Left align Alpha, Center others
        )
        print(latex_str)
        print("----------------------------")
    else:
        print("No data collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NE Table")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0], help="List of alphas")
    args = parser.parse_args()

    generate_ne_table(args.alphas)