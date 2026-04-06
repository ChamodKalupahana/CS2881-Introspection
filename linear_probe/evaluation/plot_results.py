import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import re
from pathlib import Path

# Metadata for probe categorization
# Finding project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METHOD_SCAN_DIRS = {
    "calibration": PROJECT_ROOT / "linear_probe/calibation_correct_vs_detected_correct/probe_vectors",
    "not_detected": PROJECT_ROOT / "linear_probe/not_detected_vs_detected_correct/layer_and_position_sweep/probe_vectors"
}

# Default output directory for centralized results
DEFAULT_PLOT_DIR = PROJECT_ROOT / "plots/linear_probe/evaluation"

def get_probe_method_mapping():
    """Builds a mapping of {probe_filename: method_label} by scanning project directories."""
    mapping = {}
    for method, dir_path in METHOD_SCAN_DIRS.items():
        if not dir_path.exists():
            print(f"⚠️  Warning: Method directory skip (not found): {dir_path}")
            continue
        for f in dir_path.glob("*.pt"):
            mapping[f.name] = method
    return mapping

def enrich_data(df):
    """Enriches the probe data with method categorization and formatted display names."""
    mapping = get_probe_method_mapping()
    
    # 1. Map Method
    df['Method'] = df['Probe'].apply(lambda x: mapping.get(x, "unknown"))
    
    # 2. Extract Layer Number (Lxx) for sorting
    def extract_layer(name):
        match = re.search(r"L(\d+)", name)
        return int(match.group(1)) if match else 0
    df['Layer_Num'] = df['Probe'].apply(extract_layer)
    
    # 3. Create Display Name
    method_abbr = {"calibration": "CAL", "not_detected": "ND", "unknown": "???"}
    
    def format_display_name(row):
        m = method_abbr.get(row['Method'], row['Method'][:3].upper())
        clean_name = row['Probe'].replace("MM_", "").replace(".pt", "")
        return f"[{m}] {clean_name}"
    
    df['Display_Name'] = df.apply(format_display_name, axis=1)
    
    # 4. Global Sorting Strategy: Method (CAL first) then Layer
    df = df.sort_values(by=['Method', 'Layer_Num'], ascending=[True, True])
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot aggregated linear probe evaluation results with method categorization.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to final_evaluation_summary.csv")
    parser.add_argument("--prompt", type=str, default=None, help="Filter for a specific prompt type (e.g., Anthropic)")
    parser.add_argument("--output_path", type=str, default=None, help="Target path for the PNG plot. Defaults to centralized plots folder.")
    args = parser.parse_args()

    # 1. Load Data
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV not found at {csv_path}")
        sys.exit(1)
        
    print(f"📊 Loading evaluation results from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Probe', 'Coeff'])
    
    # Optional Filtering
    if args.prompt:
        print(f"🔍 Filtering for prompt type: {args.prompt}")
        df = df[df['Prompt'] == args.prompt]
        if df.empty:
            print(f"❌ Error: No data found for prompt '{args.prompt}'")
            sys.exit(1)
    
    # 2. Dynamic Enrichment
    df = enrich_data(df)
    
    # 3. Cleanup & Processing
    df['Coeff'] = pd.to_numeric(df['Coeff'], errors='coerce')
    df['Steer_Rate'] = pd.to_numeric(df['Steer_Rate'], errors='coerce')
    df['FPR_Rate'] = pd.to_numeric(df['FPR_Rate'], errors='coerce')
    
    df_melted = df.melt(
        id_vars=['Probe', 'Display_Name', 'Method', 'Layer_Num', 'Prompt', 'Layer', 'Coeff'], 
        value_vars=['Steer_Rate', 'FPR_Rate'],
        var_name='Metric', 
        value_name='Percentage'
    )
    
    # 4. Plotting (5x2 Grid grouped by Training Method)
    print(f"📈 Generating 5x2 grid (Grouped by training approach)...")
    sns.set_theme(style="whitegrid")
    
    # Interleave logic to get CAL on left, ND on right
    cal_probes = df[df['Method'] == 'calibration']['Display_Name'].unique().tolist()
    nd_probes = df[df['Method'] == 'not_detected']['Display_Name'].unique().tolist()
    
    interleaved_order = []
    max_len = max(len(cal_probes), len(nd_probes))
    for i in range(max_len):
        if i < len(cal_probes): interleaved_order.append(cal_probes[i])
        if i < len(nd_probes): interleaved_order.append(nd_probes[i])
    
    g = sns.FacetGrid(
        df_melted, 
        col='Display_Name', 
        col_order=interleaved_order,
        col_wrap=2,
        hue='Metric',
        height=3.8, 
        aspect=2.0,
        palette={'Steer_Rate': '#44bb44', 'FPR_Rate': '#ff4444'},
        sharex=True,
        sharey=True
    )
    
    g.map(sns.lineplot, 'Coeff', 'Percentage', marker='o', markersize=5, alpha=0.9, errorbar='se')
    
    g.set_axis_labels("Coeff", "Rate (%)")
    g.set_titles(col_template="{col_name}")
    
    g.map(plt.axhline, y=0, color='gray', linestyle='--', alpha=0.4)
    g.map(plt.axhline, y=100, color='gray', linestyle='--', alpha=0.4)
    g.map(plt.axvline, x=0, color='black', alpha=0.2, linestyle='-')
    
    g.add_legend(title="Metric")
    
    plt.subplots_adjust(top=0.98, hspace=0.4, wspace=0.1)
    prompt_tag = f"({args.prompt})" if args.prompt else "Aggregated Across Prompts"
    g.fig.suptitle(f"Linear Probe Performance", fontsize=20, y=1.02)
    
    # 5. Centralized Save Extraction
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Determine Run name from input directory
        run_name = csv_path.parent.name
        # If filtered by prompt, reflect in filename
        prompt_suffix = f"_{args.prompt}" if args.prompt else "_Aggregated"
        output_filename = f"{run_name}{prompt_suffix}_evaluation.png"
        
        # Ensure centralized directory exists
        DEFAULT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_PLOT_DIR / output_filename
    
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"✅ Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
