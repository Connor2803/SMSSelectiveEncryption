import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
import glob
import argparse
from matplotlib.ticker import FuncFormatter

OUT_DIR = "plots"

def parse_time_to_seconds(s: str) -> float:
    """
    Parses time strings and converts to seconds.
    Supports: hours (h), minutes (m), seconds (s), milliseconds (ms), microseconds (µs, us)
    Examples: '1m39.5861371s', '2m9.75s', '35.2s', '123.45ms', '996.7µs', '1h2m3.5s'
    Returns total seconds as float.
    """
    if pd.isna(s):
        return float("nan")
    s = str(s).strip()
    
    # Handle microseconds (both µs and us variants)
    microsecond_pattern = r'(\d+(?:\.\d+)?)\s*([µu]s)'
    microsecond_parts = re.findall(microsecond_pattern, s)
    
    # Handle other time units (h, ms, m, s) - note: ms must come before m
    time_pattern = r'(\d+(?:\.\d+)?)\s*(h|ms|m|s)'
    time_parts = re.findall(time_pattern, s)
    
    total = 0.0
    
    # Process microseconds first
    for val, unit in microsecond_parts:
        v = float(val)
        total += v / 1_000_000  # Convert microseconds to seconds
    
    # Process other time units
    for val, unit in time_parts:
        v = float(val)
        if unit == 'h':
            total += v * 3600
        elif unit == 'm':
            total += v * 60
        elif unit == 'ms':
            total += v / 1000  # Convert milliseconds to seconds
        elif unit == 's':
            total += v
    
    # If no time units found, try to parse as plain number
    if total == 0.0 and not microsecond_parts and not time_parts:
        try:
            total = float(s)
        except Exception:
            return float("nan")
    
    return total

def extract_fuzz_percentage(filename: str, pattern: str) -> int:
    """
    Extract fuzzing percentage from filename using the provided pattern.
    Pattern should contain '#' as placeholder for the percentage number.
    E.g., 'fixedtime_performance_#.csv' matches 'fixedtime_performance_95.csv' -> 95
    """
    # Convert pattern to regex by escaping special chars and replacing # with (\d+)
    regex_pattern = re.escape(pattern).replace(r'\#', r'(\d+)')
    match = re.search(regex_pattern, filename)
    if match:
        return int(match.group(1))
    return None

def load_all_fuzz_data(file_pattern: str = "fixedtime_performance_#.csv") -> pd.DataFrame:
    """
    Load all files matching the specified pattern and combine them into one DataFrame
    with an additional 'fuzz_pct' column extracted from the filename.
    
    Args:
        file_pattern: Pattern with '#' as placeholder for percentage number
                     E.g., "fixedtime_performance_#.csv" or "fuzz_#_test.csv"
    """
    # Convert pattern to glob pattern by replacing # with *
    glob_pattern = "./" + file_pattern.replace('#', '*')
    csv_files = glob.glob(glob_pattern)
    
    # Filter to only files that match the exact pattern (with digits where # is)
    fuzz_files = []
    regex_pattern = re.escape(file_pattern).replace(r'\#', r'\d+')
    for file in csv_files:
        if re.match(f'^{regex_pattern}$', os.path.basename(file)):
            fuzz_files.append(file)
    
    if not fuzz_files:
        print(f"No CSV files found matching pattern: {file_pattern}")
        return pd.DataFrame()
    
    # Sort by the fuzzing percentage
    fuzz_files.sort(key=lambda x: extract_fuzz_percentage(os.path.basename(x), file_pattern))
    
    print(f"Found {len(fuzz_files)} fuzzing test files matching pattern '{file_pattern}':")
    for file in fuzz_files:
        fuzz_pct = extract_fuzz_percentage(os.path.basename(file), file_pattern)
        print(f"  - {os.path.basename(file)} (fuzzing {fuzz_pct}%)")
    
    # Load and combine all files
    all_data = []
    for file_path in fuzz_files:
        filename = os.path.basename(file_path)
        fuzz_pct = extract_fuzz_percentage(filename, file_pattern)
        
        if fuzz_pct is None:
            print(f"Warning: Could not extract fuzzing percentage from {filename}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            df['fuzz_pct'] = fuzz_pct
            df['source_file'] = filename
            all_data.append(df)
            print(f"  Loaded {len(df)} rows from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} total rows")
    print(f"Fuzzing percentages: {sorted(combined_df['fuzz_pct'].unique())}")
    print(f"Approaches: {sorted(combined_df['approach'].unique())}")
    print(f"Tolerances: {sorted(combined_df['tolerance'].unique())}")
    print(f"Encryption ratios: {sorted(combined_df['ratio'].unique())}")
    
    return combined_df

def make_fuzzing_asr_heatmap(df: pd.DataFrame):
    """
    Create heatmap showing ASR values with fuzzing percentage on Y-axis 
    and encryption ratio on X-axis.
    """
    # Ensure numeric types
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
    df["fuzz_pct"] = pd.to_numeric(df["fuzz_pct"], errors="coerce")
    df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
    
    # Create separate heatmaps for each tolerance level
    tolerances = sorted(df['tolerance'].unique())
    
    for tolerance in tolerances:
        df_tol = df[df['tolerance'] == tolerance].copy()
        
        if df_tol.empty:
            continue
        
        # Create pivot table: fuzz_pct vs ratio, values = ASR
        pivot = df_tol.pivot_table(index="fuzz_pct", columns="ratio", values="asr", aggfunc="mean")
        
        if pivot.empty:
            continue
        
        # Sort by fuzzing percentage (ascending - less fuzzing at top)
        pivot = pivot.sort_index(ascending=True)
        
        # Create the heatmap
        plt.figure(figsize=(14, 10))
        sns.set_style("white")
        
        # Use a color map that highlights low values (better security)
        vmax = max(0.5, pivot.max().max() * 1.1) if pivot.max().max() > 0 else 1.0
        ax = sns.heatmap(
            pivot,
            cmap="RdYlBu_r",  # Red=high ASR (bad), Blue=low ASR (good)
            vmin=0, vmax=vmax,
            annot=True,
            fmt=".4f",
            cbar_kws={"label": "Attack Success Rate (ASR)"},
            linewidths=0.5,
            linecolor='white'
        )
        
        # Create title based on tolerance
        if tolerance == 0.0:
            title = "Attack Success Rate vs Fuzzing Percentage and Encryption Ratio\n(No Fuzzing Applied - Baseline)"
            subtitle = "Selective Encryption Only"
        else:
            title = f"Attack Success Rate vs Fuzzing Percentage and Encryption Ratio\n(Tolerance = {tolerance})"
            subtitle = f"Partial Dataset Fuzzing — Adaptive (tolerance={tolerance})"
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("Encryption Ratio (%)", fontsize=14)
        plt.ylabel("Fuzzing Percentage (%)", fontsize=14)
        
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        
        # Save heatmap with tolerance in filename
        tol_str = "baseline" if tolerance == 0.0 else f"tol{tolerance:.2f}".replace(".", "")
        out = os.path.join(OUT_DIR, f"fuzzing_percentage_vs_encryption_asr_heatmap_{tol_str}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def make_fuzzing_time_heatmap(df: pd.DataFrame):
    """
    Create heatmap showing execution time with fuzzing percentage on Y-axis 
    and encryption ratio on X-axis.
    """
    # Parse time strings to seconds - handle both old and new formats
    df_time = df.copy()
    
    # Check if we have new timing format (total_time) or old format (time)
    if 'total_time' in df.columns:
        df_time["time_s"] = df_time["preprocessing_time"].apply(parse_time_to_seconds)
    elif 'time' in df.columns:
        df_time["time_s"] = df_time["time"].apply(parse_time_to_seconds)
    else:
        print("No timing data available (no 'time' or 'total_time' column found)")
        return
    
    # Ensure numeric types
    df_time["ratio"] = pd.to_numeric(df_time["ratio"], errors="coerce")
    df_time["fuzz_pct"] = pd.to_numeric(df_time["fuzz_pct"], errors="coerce")
    
    # Create pivot table: fuzz_pct vs ratio, values = time in seconds
    pivot = df_time.pivot_table(index="fuzz_pct", columns="ratio", values="time_s", aggfunc="mean")
    
    if pivot.empty:
        print("No timing data available for heatmap")
        return
    
    # Sort by fuzzing percentage (ascending)
    pivot = pivot.sort_index(ascending=True)
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    sns.set_style("white")
    
    # Use a color map for timing (lighter = faster)
    ax = sns.heatmap(
        pivot,
        cmap="YlOrRd",  # Yellow=fast, Red=slow
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Execution Time (seconds)"},
        linewidths=0.5,
        linecolor='white'
    )
    
    plt.title("Execution Time vs Fuzzing Percentage and Encryption Ratio", fontsize=16, pad=20)
    plt.xlabel("Encryption Ratio (%)", fontsize=14)
    plt.ylabel("Fuzzing Percentage (%)", fontsize=14)
    
    # Add subtitle with approach and tolerance info
    if len(df['approach'].unique()) == 1 and len(df['tolerance'].unique()) == 1:
        approach = df['approach'].iloc[0]
        tolerance = df['tolerance'].iloc[0]
        plt.suptitle(f"Partial Dataset Fuzzing Performance — {approach.capitalize()} (tolerance={tolerance})", 
                    fontsize=18, y=0.98)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Save heatmap
    out = os.path.join(OUT_DIR, "fuzzing_percentage_vs_encryption_time_heatmap.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def make_combined_analysis_plot(df: pd.DataFrame):
    """
    Create a multi-panel plot showing the relationship between fuzzing percentage,
    encryption ratio, and both ASR and execution time for each tolerance.
    """
    # Ensure numeric types
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
    df["fuzz_pct"] = pd.to_numeric(df["fuzz_pct"], errors="coerce")
    df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
    df_time = df.copy()
    
    # Check if we have new timing format (total_time) or old format (time)
    if 'total_time' in df.columns:
        df_time["time_s"] = df_time["preprocessing_time"].apply(parse_time_to_seconds)
    elif 'time' in df.columns:
        df_time["time_s"] = df_time["time"].apply(parse_time_to_seconds)
    else:
        print("No timing data available for combined analysis")
        return
    
    tolerances = sorted(df['tolerance'].unique())
    
    for tolerance in tolerances:
        df_tol = df[df['tolerance'] == tolerance].copy()
        df_time_tol = df_time[df_time['tolerance'] == tolerance].copy()
        
        if df_tol.empty:
            continue
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ASR heatmap
        pivot_asr = df_tol.pivot_table(index="fuzz_pct", columns="ratio", values="asr", aggfunc="mean")
        if not pivot_asr.empty:
            pivot_asr = pivot_asr.sort_index(ascending=True)
            vmax = max(0.5, pivot_asr.max().max() * 1.1) if pivot_asr.max().max() > 0 else 1.0
            sns.heatmap(pivot_asr, cmap="RdYlBu_r", vmin=0, vmax=vmax, 
                       annot=True, fmt=".3f", ax=ax1, cbar_kws={"label": "ASR"})
            ax1.set_title("Attack Success Rate")
            ax1.set_xlabel("Encryption Ratio (%)")
            ax1.set_ylabel("Fuzzing Percentage (%)")
        
        # 2. Time heatmap
        pivot_time = df_time_tol.pivot_table(index="fuzz_pct", columns="ratio", values="time_s", aggfunc="mean")
        if not pivot_time.empty:
            pivot_time = pivot_time.sort_index(ascending=True)
            sns.heatmap(pivot_time, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax2, 
                       cbar_kws={"label": "Time (s)"})
            ax2.set_title("Execution Time")
            ax2.set_xlabel("Encryption Ratio (%)")
            ax2.set_ylabel("Fuzzing Percentage (%)")
        
        # 3. ASR vs Fuzzing Percentage (averaged across encryption ratios)
        asr_by_fuzz = df_tol.groupby('fuzz_pct')['asr'].mean().reset_index()
        ax3.plot(asr_by_fuzz['fuzz_pct'], asr_by_fuzz['asr'], 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel("Fuzzing Percentage (%)")
        ax3.set_ylabel("Average ASR")
        ax3.set_title("ASR vs Fuzzing Percentage\n(averaged across all encryption ratios)")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(1.0, asr_by_fuzz['asr'].max() * 1.1))
        
        # 4. Time vs Fuzzing Percentage (averaged across encryption ratios)
        time_by_fuzz = df_time_tol.groupby('fuzz_pct')['time_s'].mean().reset_index()
        ax4.plot(time_by_fuzz['fuzz_pct'], time_by_fuzz['time_s'], 'ro-', linewidth=2, markersize=8)
        ax4.set_xlabel("Fuzzing Percentage (%)")
        ax4.set_ylabel("Average Time (seconds)")
        ax4.set_title("Execution Time vs Fuzzing Percentage\n(averaged across all encryption ratios)")
        ax4.grid(True, alpha=0.3)
        
        # Add overall title
        approach = df_tol['approach'].iloc[0] if len(df_tol['approach'].unique()) == 1 else "Mixed"
        if tolerance == 0.0:
            fig.suptitle(f"Baseline Analysis — {approach.capitalize()} (No Fuzzing)", 
                        fontsize=18, y=0.98)
        else:
            fig.suptitle(f"Partial Dataset Fuzzing Analysis — {approach.capitalize()} (tolerance={tolerance})", 
                        fontsize=18, y=0.98)
        
        plt.tight_layout()
        
        # Save combined plot with tolerance in filename
        tol_str = "baseline" if tolerance == 0.0 else f"tol{tolerance:.2f}".replace(".", "")
        out = os.path.join(OUT_DIR, f"fuzzing_percentage_combined_analysis_{tol_str}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def make_fuzzing_comparison_plot(df: pd.DataFrame):
    """
    Create a side-by-side comparison of baseline (no fuzzing) vs fuzzing effect.
    """
    # Ensure numeric types
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
    df["fuzz_pct"] = pd.to_numeric(df["fuzz_pct"], errors="coerce")
    df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
    
    # Get baseline data (tolerance = 0.0) and fuzzing data (tolerance > 0.0)
    baseline_df = df[df['tolerance'] == 0.0].copy()
    fuzzing_df = df[df['tolerance'] > 0.0].copy()
    
    if baseline_df.empty or fuzzing_df.empty:
        print("Cannot create comparison plot: missing baseline or fuzzing data")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # 1. Baseline ASR heatmap
    pivot_baseline = baseline_df.pivot_table(index="fuzz_pct", columns="ratio", values="asr", aggfunc="mean")
    if not pivot_baseline.empty:
        pivot_baseline = pivot_baseline.sort_index(ascending=True)
        sns.heatmap(pivot_baseline, cmap="RdYlBu_r", vmin=0, vmax=1.0, 
                   annot=True, fmt=".3f", ax=ax1, cbar_kws={"label": "ASR"})
        ax1.set_title("Baseline (No Fuzzing)\nSelective Encryption Only")
        ax1.set_xlabel("Encryption Ratio (%)")
        ax1.set_ylabel("Fuzzing Percentage (%)")
    
    # 2. Fuzzing ASR heatmap
    pivot_fuzzing = fuzzing_df.pivot_table(index="fuzz_pct", columns="ratio", values="asr", aggfunc="mean")
    if not pivot_fuzzing.empty:
        pivot_fuzzing = pivot_fuzzing.sort_index(ascending=True)
        tolerance_val = fuzzing_df['tolerance'].iloc[0]
        sns.heatmap(pivot_fuzzing, cmap="RdYlBu_r", vmin=0, vmax=1.0, 
                   annot=True, fmt=".3f", ax=ax2, cbar_kws={"label": "ASR"})
        ax2.set_title(f"With Fuzzing (tolerance={tolerance_val})\nPartial Dataset Fuzzing")
        ax2.set_xlabel("Encryption Ratio (%)")
        ax2.set_ylabel("Fuzzing Percentage (%)")
    
    # 3. ASR Reduction (Baseline - Fuzzing)
    if not pivot_baseline.empty and not pivot_fuzzing.empty:
        # Align the indices and columns
        common_index = pivot_baseline.index.intersection(pivot_fuzzing.index)
        common_columns = pivot_baseline.columns.intersection(pivot_fuzzing.columns)
        
        if len(common_index) > 0 and len(common_columns) > 0:
            baseline_aligned = pivot_baseline.loc[common_index, common_columns]
            fuzzing_aligned = pivot_fuzzing.loc[common_index, common_columns]
            
            asr_reduction = baseline_aligned - fuzzing_aligned
            
            # Use a diverging colormap: blue = more reduction (good), red = less reduction (bad)
            vmax = max(abs(asr_reduction.min().min()), abs(asr_reduction.max().max()))
            sns.heatmap(asr_reduction, cmap="RdBu", vmin=-vmax, vmax=vmax, center=0,
                       annot=True, fmt=".3f", ax=ax3, cbar_kws={"label": "ASR Reduction"})
            ax3.set_title("ASR Reduction\n(Baseline - Fuzzing)")
            ax3.set_xlabel("Encryption Ratio (%)")
            ax3.set_ylabel("Fuzzing Percentage (%)")
    
    plt.suptitle("Partial Dataset Fuzzing: Baseline vs Fuzzing Comparison", fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Save comparison plot
    out = os.path.join(OUT_DIR, "fuzzing_baseline_vs_fuzzing_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def make_asr_vs_fuzzing_plots(df: pd.DataFrame):
    """
    Create separate plots showing ASR vs Fuzzing Percentage:
    1. Average across all encryption ratios
    2. Specific to 0% encryption ratio (no encryption)
    """
    # Ensure numeric types
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
    df["fuzz_pct"] = pd.to_numeric(df["fuzz_pct"], errors="coerce")
    df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
    
    tolerances = sorted(df['tolerance'].unique())
    
    for tolerance in tolerances:
        df_tol = df[df['tolerance'] == tolerance].copy()
        
        if df_tol.empty:
            continue
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. ASR vs Fuzzing Percentage (separate line for each encryption ratio)
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_tol['ratio'].unique())))
        max_asr = 0
        
        for i, ratio in enumerate(sorted(df_tol['ratio'].unique())):
            df_ratio = df_tol[df_tol['ratio'] == ratio].copy()
            if not df_ratio.empty:
                asr_by_fuzz = df_ratio.groupby('fuzz_pct')['asr'].mean().reset_index()
                ax1.plot(asr_by_fuzz['fuzz_pct'], asr_by_fuzz['asr'], 
                        'o-', color=colors[i], linewidth=2, markersize=6, 
                        alpha=0.8, label=f'{int(ratio)}% encryption')
                max_asr = max(max_asr, asr_by_fuzz['asr'].max())
        
        ax1.set_xlabel("Fuzzing Percentage (%)", fontsize=12)
        ax1.set_ylabel("ASR", fontsize=12)
        ax1.set_title("ASR vs Fuzzing Percentage\n(Each encryption ratio)", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, max(1.0, max_asr * 1.1))
        
        if not df_tol.empty:
            ax1.set_xlim(df_tol['fuzz_pct'].min() - 2, df_tol['fuzz_pct'].max() + 2)
        
        # 2. ASR vs Fuzzing Percentage (0% encryption ratio only)
        df_0_ratio = df_tol[df_tol['ratio'] == 0].copy()
        if not df_0_ratio.empty:
            asr_by_fuzz_0 = df_0_ratio.groupby('fuzz_pct')['asr'].mean().reset_index()
            ax2.plot(asr_by_fuzz_0['fuzz_pct'], asr_by_fuzz_0['asr'], 'go-', linewidth=3, markersize=10, alpha=0.8)
            ax2.set_xlabel("Fuzzing Percentage (%)", fontsize=12)
            ax2.set_ylabel("ASR (0% Encryption)", fontsize=12)
            ax2.set_title("ASR vs Fuzzing Percentage\n(0% Encryption Ratio)", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(1.0, asr_by_fuzz_0['asr'].max() * 1.1))
            ax2.set_xlim(asr_by_fuzz_0['fuzz_pct'].min() - 2, asr_by_fuzz_0['fuzz_pct'].max() + 2)
            
        else:
            ax2.text(0.5, 0.5, 'No data for 0% encryption ratio', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_xlabel("Fuzzing Percentage (%)", fontsize=12)
            ax2.set_ylabel("ASR (0% Encryption)", fontsize=12)
            ax2.set_title("ASR vs Fuzzing Percentage\n(0% Encryption Ratio - No Selective Encryption)", fontsize=14)
        
        # Add overall title
        approach = df_tol['approach'].iloc[0] if len(df_tol['approach'].unique()) == 1 else "Mixed"
        if tolerance == 0.0:
            fig.suptitle(f"Baseline Analysis — {approach.capitalize()} (No Fuzzing)", fontsize=16, y=1.02)
        else:
            fig.suptitle(f"ASR vs Fuzzing Percentage — {approach.capitalize()} (T={tolerance})", fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        # Save plot with tolerance in filename
        tol_str = "baseline" if tolerance == 0.0 else f"tol{tolerance:.2f}".replace(".", "")
        out = os.path.join(OUT_DIR, f"asr_vs_fuzzing_percentage_{tol_str}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def make_preprocessing_time_analysis(df: pd.DataFrame):
    """
    Create plot showing average preprocessing time vs fuzzing percentage (across all encryption ratios).
    Also print the raw averages for 0% and 100% fuzzing.
    """
    # Check if preprocessing_time column exists (new timing format)
    if 'preprocessing_time' not in df.columns:
        print("No preprocessing_time column found. Skipping preprocessing time analysis.")
        return

    # Parse preprocessing time strings to seconds
    df_time = df.copy()
    df_time["preprocessing_time_s"] = df_time["preprocessing_time"].apply(parse_time_to_seconds)

    # Ensure numeric types
    df_time["ratio"] = pd.to_numeric(df_time["ratio"], errors="coerce")
    df_time["fuzz_pct"] = pd.to_numeric(df_time["fuzz_pct"], errors="coerce")
    df_time["tolerance"] = pd.to_numeric(df_time["tolerance"], errors="coerce")

    # Remove rows with invalid preprocessing times
    df_time = df_time.dropna(subset=['preprocessing_time_s'])

    if df_time.empty:
        print("No valid preprocessing time data found.")
        return

    tolerances = sorted(df_time['tolerance'].unique())

    for tolerance in tolerances:
        df_tol = df_time[df_time['tolerance'] == tolerance].copy()

        if df_tol.empty:
            continue

        # Print raw averages for 0% and 100% fuzzing
        for pct in [0, 100]:
            avg = df_tol[df_tol['fuzz_pct'] == pct]['preprocessing_time_s'].mean()
            if not np.isnan(avg):
                print(f"Average preprocessing time for fuzz_pct={pct}% (tolerance={tolerance}): {avg:.6f} seconds ({avg*1000:.2f} ms)")
            else:
                print(f"No data for fuzz_pct={pct}% (tolerance={tolerance})")

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Average preprocessing time vs fuzzing percentage (across all encryption ratios)
        preprocessing_by_fuzz = df_tol.groupby('fuzz_pct')['preprocessing_time_s'].mean().reset_index()
        ax.plot(preprocessing_by_fuzz['fuzz_pct'], preprocessing_by_fuzz['preprocessing_time_s'],
                'bo-', linewidth=3, markersize=8, alpha=0.8)
        ax.set_xlabel("Fuzzing Percentage (%)", fontsize=12)
        ax.set_ylabel("Average Preprocessing Time (seconds)", fontsize=12)
        ax.set_title("Preprocessing Time vs Fuzzing Percentage\n(Average across all encryption ratios)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, preprocessing_by_fuzz['preprocessing_time_s'].max() * 1.1)
        ax.set_xlim(preprocessing_by_fuzz['fuzz_pct'].min() - 2, preprocessing_by_fuzz['fuzz_pct'].max() + 2)

        # Add overall title
        approach = df_tol['approach'].iloc[0] if len(df_tol['approach'].unique()) == 1 else "Mixed"
        if tolerance == 0.0:
            fig.suptitle(f"Preprocessing Time Analysis — {approach.capitalize()} (Baseline)", fontsize=16, y=1.02)
        else:
            fig.suptitle(f"Preprocessing Time Analysis — {approach.capitalize()} (T={tolerance})", fontsize=16, y=1.02)

        plt.tight_layout()

        # Save plot with tolerance in filename
        tol_str = "baseline" if tolerance == 0.0 else f"tol{tolerance:.2f}".replace(".", "")
        out = os.path.join(OUT_DIR, f"preprocessing_time_vs_fuzzing_percentage_{tol_str}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def make_encryption_time_analysis(df: pd.DataFrame):
    """
    Create plot showing average encryption time vs fuzzing percentage (across all encryption ratios).
    Also print the raw averages for 0% and 100% fuzzing.
    """
    # Check if encryption_time column exists (new timing format)
    if 'encryption_time' not in df.columns:
        print("No encryption_time column found. Skipping encryption time analysis.")
        return

    # Parse encryption time strings to seconds
    df_time = df.copy()
    df_time["encryption_time_s"] = df_time["encryption_time"].apply(parse_time_to_seconds)

    # Ensure numeric types
    df_time["ratio"] = pd.to_numeric(df_time["ratio"], errors="coerce")
    df_time["fuzz_pct"] = pd.to_numeric(df_time["fuzz_pct"], errors="coerce")
    df_time["tolerance"] = pd.to_numeric(df_time["tolerance"], errors="coerce")

    # Remove rows with invalid encryption times
    df_time = df_time.dropna(subset=['encryption_time_s'])

    if df_time.empty:
        print("No valid encryption time data found.")
        return

    tolerances = sorted(df_time['tolerance'].unique())

    for tolerance in tolerances:
        df_tol = df_time[df_time['tolerance'] == tolerance].copy()

        if df_tol.empty:
            continue

        # Print raw averages for all fuzzing percentages
        print(f"\n=== Encryption Time Analysis (tolerance={tolerance}) ===")
        for pct in sorted(df_tol['fuzz_pct'].unique()):
            avg = df_tol[df_tol['fuzz_pct'] == pct]['encryption_time_s'].mean()
            if not np.isnan(avg):
                if avg < 1:
                    print(f"Average encryption time for fuzz_pct={pct:3d}% (tolerance={tolerance}): {avg:.6f} seconds ({avg*1000:.2f} ms)")
                else:
                    minutes = int(avg // 60)
                    seconds = avg % 60
                    if minutes > 0:
                        print(f"Average encryption time for fuzz_pct={pct:3d}% (tolerance={tolerance}): {avg:.2f} seconds ({minutes}m{seconds:.2f}s)")
                    else:
                        print(f"Average encryption time for fuzz_pct={pct:3d}% (tolerance={tolerance}): {avg:.2f} seconds")
            else:
                print(f"No data for fuzz_pct={pct}% (tolerance={tolerance})")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Average encryption time vs fuzzing percentage (across all encryption ratios)
        encryption_by_fuzz = df_tol.groupby('fuzz_pct')['encryption_time_s'].mean().reset_index()
        ax.plot(encryption_by_fuzz['fuzz_pct'], encryption_by_fuzz['encryption_time_s'],
                'ro-', linewidth=3, markersize=8, alpha=0.8)
        ax.set_xlabel("Fuzzing Percentage (%)", fontsize=12)
        ax.set_ylabel("Average Encryption Time (seconds)", fontsize=12)
        ax.set_title("Encryption Time vs Fuzzing Percentage\n(Average across all encryption ratios)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, encryption_by_fuzz['encryption_time_s'].max() * 1.1)
        ax.set_xlim(encryption_by_fuzz['fuzz_pct'].min() - 2, encryption_by_fuzz['fuzz_pct'].max() + 2)

        # Format y-axis to show time in minutes if > 60 seconds
        def format_time(x, p):
            if x >= 60:
                minutes = int(x // 60)
                seconds = x % 60
                return f"{minutes}m{seconds:.0f}s"
            else:
                return f"{x:.1f}s"
        
        ax.yaxis.set_major_formatter(FuncFormatter(format_time))

        # Add overall title
        approach = df_tol['approach'].iloc[0] if len(df_tol['approach'].unique()) == 1 else "Mixed"
        if tolerance == 0.0:
            fig.suptitle(f"Encryption Time Analysis — {approach.capitalize()} (Baseline)", fontsize=16, y=1.02)
        else:
            fig.suptitle(f"Encryption Time Analysis — {approach.capitalize()} (T={tolerance})", fontsize=16, y=1.02)

        plt.tight_layout()

        # Save plot with tolerance in filename
        tol_str = "baseline" if tolerance == 0.0 else f"tol{tolerance:.2f}".replace(".", "")
        out = os.path.join(OUT_DIR, f"encryption_time_vs_fuzzing_percentage_{tol_str}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate fuzzing analysis plots from CSV files')
    parser.add_argument('-p', '--pattern', 
                       default='fixedtime_performance_#.csv',
                       help='File pattern with # as placeholder for percentage number (default: fixedtime_performance_#.csv)')
    parser.add_argument('-o', '--output', 
                       default='plots',
                       help='Output directory for plots (default: plots)')
    
    args = parser.parse_args()
    
    # Update global output directory
    global OUT_DIR
    OUT_DIR = args.output
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=== Fuzzing Percentage Analysis ===")
    print(f"File pattern: {args.pattern}")
    print(f"Output directory: {OUT_DIR}")
    
    # Load all fuzzing test data
    df = load_all_fuzz_data(args.pattern)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    print(f"\nDataset summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Fuzzing percentages: {sorted(df['fuzz_pct'].unique())}")
    print(f"  Encryption ratios: {sorted(df['ratio'].unique())}")
    print(f"  Approaches: {list(df['approach'].unique())}")
    print(f"  Tolerances: {list(df['tolerance'].unique())}")
    
    # Generate all heatmaps and analysis plots
    print("\nGenerating visualizations...")
    make_fuzzing_asr_heatmap(df)
    make_fuzzing_time_heatmap(df)
    make_combined_analysis_plot(df)
    make_fuzzing_comparison_plot(df)
    make_asr_vs_fuzzing_plots(df)
    make_preprocessing_time_analysis(df)
    make_encryption_time_analysis(df)
    
    print(f"\n✓ All fuzzing analysis plots saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()