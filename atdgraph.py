import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Output directory
OUT_DIR = "plots/individual_atd"

def parse_time_to_seconds(s: str) -> float:
    """
    Parses strings like '1m39.5861371s', '2m9.75s', '35.2s', optionally with hours '1h2m3.5s'.
    Returns total seconds as float.
    """
    if pd.isna(s):
        return float("nan")
    s = str(s).strip()
    parts = re.findall(r'(\d+(?:\.\d+)?)\s*([hms])', s)
    total = 0.0
    for val, unit in parts:
        v = float(val)
        if unit == 'h':
            total += v * 3600.0
        elif unit == 'm':
            total += v * 60.0
        elif unit == 's':
            total += v
    if total == 0.0 and not parts:
        try:
            total = pd.to_timedelta(s).total_seconds()
        except Exception:
            pass
    return total

def create_method_atd_graphs():
    """Create one big graph for each method showing ATD vs ASR for different ratios"""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    approaches = ['adaptive', 'sliding']
    atd_sizes = list(range(3, 25, 3))  # 3, 6, 9, 12, 15, 18, 21, 24
    
    for approach in approaches:
        # Collect all data for this approach
        all_data = []
        
        for atd_size in atd_sizes:
            filename = f"{approach}{atd_size}.csv"
            
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found, skipping...")
                continue
            
            try:
                df = pd.read_csv(filename)
                df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
                df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
                df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
                df["atd_size"] = atd_size
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        if not all_data:
            print(f"No data found for {approach} method")
            continue
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create the big graph for this method
        create_atd_vs_asr_graph(combined_df, approach)
        
        print(f"✓ Created ATD graph for {approach} method")

def create_atd_vs_asr_graph(df, approach):
    """Create ATD vs ASR graph for one method with different ratio lines"""
    
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    # Get unique ratios and sort them
    ratios = sorted(df['ratio'].unique())
    tolerance = df['tolerance'].iloc[0]
    
    # Color palette for different ratios
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))
    
    # Plot each ratio as a separate line
    for i, ratio in enumerate(ratios):
        ratio_data = df[df['ratio'] == ratio].sort_values('atd_size')
        
        plt.plot(ratio_data['atd_size'], ratio_data['asr'], 
                'o-', linewidth=3, markersize=8, 
                color=colors[i], label=f'{int(ratio)}% Encryption',
                alpha=0.8)
        
        # Add value annotations for key points
        for _, row in ratio_data.iterrows():
            if row['atd_size'] in [3, 12, 24]:  # Annotate key ATD sizes
                plt.annotate(f'{row["asr"]:.3f}', 
                            (row['atd_size'], row['asr']),
                            textcoords="offset points", 
                            xytext=(0, 8), ha='center', 
                            fontsize=8, alpha=0.7)
    
    plt.xlabel("ATD Size (Attacker Time Delta)", fontsize=14)
    plt.ylabel("Attack Success Rate (ASR)", fontsize=14)
    plt.title(f"{approach.capitalize()} Method - ASR vs ATD Size\n(Tolerance = {tolerance})", 
              fontsize=16, fontweight='bold')
    
    # Customize the plot
    plt.xlim(2, 25)
    plt.ylim(0, 1.05)
    plt.xticks(sorted(df['atd_size'].unique()))
    plt.grid(True, alpha=0.3)
    
    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Add summary text box
    summary_text = f"""
    Method: {approach.capitalize()}
    ATD Range: {df['atd_size'].min()}-{df['atd_size'].max()}
    Encryption Range: {df['ratio'].min():.0f}-{df['ratio'].max():.0f}%
    """
    
    plt.text(0.02, 0.98, summary_text.strip(), transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", 
                      facecolor='lightgray' if approach == 'sliding' else 'lightblue', 
                      alpha=0.3))
    
    plt.tight_layout()
    output_file = os.path.join(OUT_DIR, f"{approach}_atd_vs_asr_overview.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

def create_combined_comparison():
    """Create a combined comparison showing both methods on the same graph"""
    approaches = ['adaptive', 'sliding']
    atd_sizes = list(range(3, 25, 3))
    
    # Collect data for both approaches
    method_data = {}
    
    for approach in approaches:
        all_data = []
        
        for atd_size in atd_sizes:
            filename = f"{approach}{atd_size}.csv"
            
            if not os.path.exists(filename):
                continue
            
            try:
                df = pd.read_csv(filename)
                df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
                df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
                df["atd_size"] = atd_size
                all_data.append(df)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        if all_data:
            method_data[approach] = pd.concat(all_data, ignore_index=True)
    
    if len(method_data) < 2:
        print("Not enough data for combined comparison")
        return
    
    # Create combined plot for key encryption ratios
    key_ratios = [0, 20, 50, 80]  # Focus on key ratios for clarity
    
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")
    
    method_colors = {'adaptive': 'darkorange', 'sliding': 'steelblue'}
    line_styles = ['-', '--', '-.', ':']
    
    for approach, data in method_data.items():
        tolerance = data['tolerance'].iloc[0]
        
        for i, ratio in enumerate(key_ratios):
            if ratio not in data['ratio'].values:
                continue
                
            ratio_data = data[data['ratio'] == ratio].sort_values('atd_size')
            
            plt.plot(ratio_data['atd_size'], ratio_data['asr'],
                    linestyle=line_styles[i], linewidth=3, markersize=8,
                    color=method_colors[approach], marker='o',
                    label=f'{approach.capitalize()} - {int(ratio)}%',
                    alpha=0.8)
    
    plt.xlabel("ATD Size (Attacker Time Delta)", fontsize=14)
    plt.ylabel("Attack Success Rate (ASR)", fontsize=14)
    plt.title("Method Comparison - ASR vs ATD Size for Key Encryption Ratios", 
              fontsize=16, fontweight='bold')
    
    plt.xlim(2, 25)
    plt.ylim(0, 1.05)
    plt.xticks(atd_sizes)
    plt.grid(True, alpha=0.3)
    
    # Legend with two columns
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, ncol=1)
    
    plt.tight_layout()
    output_file = os.path.join(OUT_DIR, "methods_comparison_atd_vs_asr.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Created combined methods comparison: {output_file}")

def main():
    """Main function to create ATD-focused graphs"""
    print("Creating ATD-focused graphs...")
    
    # Import numpy for color palette
    import numpy as np
    globals()['np'] = np
    
    # Create individual method graphs
    create_method_atd_graphs()
    
    # Create combined comparison
    create_combined_comparison()
    
    print(f"\n✓ All graphs saved to: {OUT_DIR}/")
    print("Generated graphs:")
    print("  - adaptive_atd_vs_asr_overview.png")
    print("  - sliding_atd_vs_asr_overview.png")
    print("  - methods_comparison_atd_vs_asr.png")

if __name__ == "__main__":
    main()