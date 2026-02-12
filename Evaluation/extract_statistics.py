import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob

def get_all_summary_data(results_path):
    """Loads all summary.csv files from alpha directories."""
    all_data = []
    alpha_dirs = glob.glob(os.path.join(results_path, "alpha_*"))
    for folder in alpha_dirs:
        try:
            # Check if it's a directory and contains summary.csv
            alpha_val = float(folder.split('_')[-1])
            summary_file = os.path.join(folder, "summary.csv")
            if os.path.exists(summary_file):
                df = pd.read_csv(summary_file)
                df['alpha'] = alpha_val
                all_data.append(df)
        except (ValueError, IndexError):
            continue
    return all_data

def generate_summary_table(all_data, results_path):
    """Calculates Mean ± SD for performance metrics and exports to CSV."""
    summary_list = []
    for df in all_data:
        alpha_val = df['alpha'].iloc[0]
        cap_mean, cap_std = df['total_captures'].mean(), df['total_captures'].std()
        coll_mean, coll_std = df['total_collisions'].mean(), df['total_collisions'].std()
        
        summary_list.append({
            "alpha Profile": alpha_val,
            "Capture Frequency": f"{cap_mean:.2f} ± {cap_std:.2f}",
            "Inter-agent Collisions": f"{coll_mean:.2f} ± {coll_std:.2f}"
        })

    final_table_df = pd.DataFrame(summary_list).sort_values(by="alpha Profile", ascending=False)
    table_path = os.path.join(results_path, "final_performance_table.csv")
    final_table_df.to_csv(table_path, index=False)
    print(f"✅ Summary Table saved to: {table_path}")

def create_clean_boxplot(all_data, results_path):
    """Generates a clean, single-color Boxplot for Capture Frequency."""
    if not all_data: return
    
    full_df = pd.concat(all_data).sort_values(by="alpha", ascending=False)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    
    # Single color 'steelblue' for all boxes
    sns.boxplot(
        x='alpha', 
        y='total_captures', 
        data=full_df, 
        color="steelblue", 
        width=0.6
    )
    
    # Add points to show distribution
    sns.stripplot(x='alpha', y='total_captures', data=full_df, color=".25", size=4, alpha=0.4)
    
    plt.xlabel(r'Alpha Coefficient ($\alpha$)', fontsize=12)
    plt.ylabel('Total Captures per Episode', fontsize=12)
    
    output_path = os.path.join(results_path, "capture_boxplot_clean.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Clean Boxplot saved to: {output_path}")

def create_combined_heatmaps(results_path):
    """Generates multi-agent trajectory heatmaps for Alpha 0.0, 0.5, and 1.0."""
    target_alphas = [0.0, 0.5, 1.0]
    # Color mapping for each agent
    agent_cmaps = {
        "agent_0": "Blues",      # Prey
        "adversary_0": "Reds",   # Predator 1
        "adversary_1": "Greens", # Predator 2
        "adversary_2": "Purples" # Predator 3
    }

    for alpha in target_alphas:
        alpha_folder = os.path.join(results_path, f"alpha_{alpha}")
        trajectory_file = os.path.join(alpha_folder, "trajectories.csv")
        
        if not os.path.exists(trajectory_file):
            print(f"⚠️ Trajectory data missing for Alpha {alpha}, skipping heatmap.")
            continue

        df_traj = pd.read_csv(trajectory_file)
        plt.figure(figsize=(10, 8))
        
        for agent_name, cmap in agent_cmaps.items():
            agent_data = df_traj[df_traj['agent name'] == agent_name]
            if agent_data.empty: continue
                
            plt.hexbin(
                agent_data['x coordinate'], 
                agent_data['y coordinate'], 
                gridsize=35, 
                cmap=cmap, 
                mincnt=1, 
                alpha=0.5, 
                edgecolors='none'
            )

        plt.title(f'Multi-Agent Trajectory Density (Alpha={alpha})', fontsize=15)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='Prey (agent_0)'),
            Line2D([0], [0], color='red', lw=4, label='Predator 1'),
            Line2D([0], [0], color='green', lw=4, label='Predator 2'),
            Line2D([0], [0], color='purple', lw=4, label='Predator 3')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        output_path = os.path.join(results_path, f"heatmap_alpha_{alpha}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Combined Heatmap saved to: {output_path}")

if __name__ == "__main__":
    BASE_RESULTS = 'results'
    
    # 1. Load Data
    summary_data = get_all_summary_data(BASE_RESULTS)
    
    if summary_data:
        # 2. Process Table
        generate_summary_table(summary_data, BASE_RESULTS)
        
        # 3. Process Boxplot (Captures only, Clean)
        create_clean_boxplot(summary_data, BASE_RESULTS)
        
        # 4. Process Heatmaps (Combined agents, Selected Alphas)
        create_combined_heatmaps(BASE_RESULTS)
        
        print("\n✨ All results generated successfully in the 'results' folder.")
    else:
        print("❌ Error: No data found. Make sure to run the evaluation first.")