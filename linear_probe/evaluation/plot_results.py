import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot aggregated linear probe evaluation results from a CSV summary.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to final_evaluation_summary.csv")
    args = parser.parse_args()

    # 1. Load Data
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV not found at {csv_path}")
        sys.exit(1)
        
    print(f"📊 Loading evaluation results from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Probe', 'Coeff'])
    
    # 2. Cleanup & Processing
    # Ensure numeric types
    df['Coeff'] = pd.to_numeric(df['Coeff'], errors='coerce')
    df['Steer_Rate'] = pd.to_numeric(df['Steer_Rate'], errors='coerce')
    df['FPR_Rate'] = pd.to_numeric(df['FPR_Rate'], errors='coerce')
    
    # Melt for dual-line plotting (Steer_Rate vs FPR_Rate)
    df_melted = df.melt(
        id_vars=['Probe', 'Prompt', 'Layer', 'Coeff'], 
        value_vars=['Steer_Rate', 'FPR_Rate'],
        var_name='Metric', 
        value_name='Percentage'
    )
    
    # 3. Optimized Visualization logic (2x5 Grid)
    print(f"📈 Generating 2x5 wrapped grid plots...")
    sns.set_theme(style="whitegrid")
    
    # Using col_wrap=5 to force a 2x5 grid if there are 10 probes
    g = sns.FacetGrid(
        df_melted, 
        col='Probe', 
        col_wrap=5,
        hue='Metric',
        height=3.5, 
        aspect=1.2,
        palette={'Steer_Rate': '#44bb44', 'FPR_Rate': '#ff4444'},
        sharex=True,
        sharey=True
    )
    
    # Plot lineplots. 
    g.map(sns.lineplot, 'Coeff', 'Percentage', marker='o', markersize=5, alpha=0.9, errorbar='se')
    
    # Formatting
    g.set_axis_labels("Coeff", "Rate (%)")
    g.set_titles(col_template="{col_name}")
    
    # Core reference lines
    g.map(plt.axhline, y=0, color='gray', linestyle='--', alpha=0.4)
    g.map(plt.axhline, y=100, color='gray', linestyle='--', alpha=0.4)
    g.map(plt.axvline, x=0, color='black', alpha=0.2, linestyle='-')
    
    # Legend
    g.add_legend(title="Performance Category")
    
    # Layout and Title
    plt.subplots_adjust(top=0.9, hspace=0.35, wspace=0.1)
    g.fig.suptitle(f"Linear Probe Performance Comparison (Wrapped Summary) | Run: {csv_path.parent.name}", fontsize=20, y=1.02)
    
    # 4. Save
    output_path = csv_path.parent / "evaluation_plot_aggregated.png"
    # bbox_inches='tight' is critical here as titles can be long
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"✅ Success! Aggregated grid plot saved to: {output_path}")

if __name__ == "__main__":
    main()
