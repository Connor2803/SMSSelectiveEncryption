import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import glob
from matplotlib.ticker import FuncFormatter

OUT_DIR = "plots"

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
            total += v * 3600
        elif unit == 'm':
            total += v * 60
        elif unit == 's':
            total += v
    if total == 0.0 and not parts:
        try:
            total = float(s)
        except Exception:
            return float("nan")
    return total

def make_raw_asr_heatmap(df: pd.DataFrame, filename: str):
    """
    Create heatmap showing raw ASR values for all methods, with baseline at the top.
    """
    # Prepare data for all approaches
    all_data = df.copy()
    all_data["ratio"] = pd.to_numeric(all_data["ratio"], errors="coerce")
    all_data["asr"] = pd.to_numeric(all_data["asr"], errors="coerce")
    all_data["tolerance"] = pd.to_numeric(all_data["tolerance"], errors="coerce")
    
    # Create method-tolerance combinations
    all_data["method_tol"] = all_data.apply(lambda row: 
        f"Baseline (SE only)" if row["approach"] == "none" 
        else f"{row['approach'].capitalize()} (tol={row['tolerance']:.2f})", axis=1)
    
    # Create pivot table
    pivot = all_data.pivot_table(index="method_tol", columns="ratio", values="asr", aggfunc="mean")
    
    # Sort to put baseline at top, then sliding, then adaptive
    baseline_rows = [idx for idx in pivot.index if "Baseline" in idx]
    sliding_rows = sorted([idx for idx in pivot.index if "Sliding" in idx])
    adaptive_rows = sorted([idx for idx in pivot.index if "Adaptive" in idx])
    
    ordered_index = baseline_rows + sliding_rows + adaptive_rows
    if ordered_index:  # Only reindex if we have rows to reorder
        pivot = pivot.reindex(ordered_index)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.set_style("white")
    
    # Use a color map that highlights low values (better security)
    ax = sns.heatmap(
        pivot,
        cmap="RdYlBu_r",  # Red=high ASR (bad), Blue=low ASR (good)
        vmin=0, vmax=1,   # ASR range
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Attack Success Rate (ASR)"},
        linewidths=0.5,
        linecolor='white'
    )
    
    # Extract the number from filename for the title
    match = re.search(r'(\d+)fulldata\.csv', filename)
    file_num = match.group(1) if match else "Unknown"
    
    plt.title(f"Raw ASR Values — {file_num}% Matching — All Methods and Tolerances", fontsize=16, pad=20)
    plt.xlabel("Encryption Ratio (%)", fontsize=12)
    plt.ylabel("Method (Tolerance)", fontsize=12)
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Save with dataset number in filename
    base_name = os.path.splitext(filename)[0]  # Remove .csv
    out = os.path.join(OUT_DIR, f"{base_name}_raw_asr_heatmap.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def make_raw_asr_heatmap_separate(df: pd.DataFrame, filename: str, approach: str):
    """
    Create separate heatmap for one approach with baseline comparison.
    """
    # Filter data for this approach + baseline
    method_data = df[df["approach"].isin([approach, "none"])].copy()
    if method_data.empty:
        print(f"No {approach} or baseline data found in {filename}; skipping separate heatmap.")
        return
    
    method_data["ratio"] = pd.to_numeric(method_data["ratio"], errors="coerce")
    method_data["asr"] = pd.to_numeric(method_data["asr"], errors="coerce")
    method_data["tolerance"] = pd.to_numeric(method_data["tolerance"], errors="coerce")
    
    # Create method-tolerance combinations
    method_data["method_tol"] = method_data.apply(lambda row: 
        f"Baseline (SE only)" if row["approach"] == "none" 
        else f"{row['approach'].capitalize()} (tol={row['tolerance']:.2f})", axis=1)
    
    # Create pivot table
    pivot = method_data.pivot_table(index="method_tol", columns="ratio", values="asr", aggfunc="mean")
    
    # Sort to put baseline at top
    baseline_rows = [idx for idx in pivot.index if "Baseline" in idx]
    method_rows = sorted([idx for idx in pivot.index if "Baseline" not in idx])
    
    ordered_index = baseline_rows + method_rows
    if ordered_index:
        pivot = pivot.reindex(ordered_index)
    
    # Create the heatmap
    plt.figure(figsize=(12, max(6, len(pivot.index) * 0.6)))
    sns.set_style("white")
    
    ax = sns.heatmap(
        pivot,
        cmap="RdYlBu_r",
        vmin=0, vmax=1,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Attack Success Rate (ASR)"},
        linewidths=0.5,
        linecolor='white'
    )
    
    # Extract the number from filename for the title
    match = re.search(r'(\d+)fulldata\.csv', filename)
    file_num = match.group(1) if match else "Unknown"
    
    plt.title(f"Raw ASR Values — {file_num}% Matching — {approach.capitalize()} vs Baseline", fontsize=16, pad=20)
    plt.xlabel("Encryption Ratio (%)", fontsize=12)
    plt.ylabel("Method (Tolerance)", fontsize=12)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Save with dataset number and approach in filename
    base_name = os.path.splitext(filename)[0]  # Remove .csv
    out = os.path.join(OUT_DIR, f"{base_name}_raw_asr_heatmap_{approach}_vs_baseline.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def _baseline_by_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ratio, base_asr, base_time_s for approach=='none'
    """
    base = df[df["approach"] == "none"].copy()
    if base.empty:
        return pd.DataFrame(columns=["ratio", "base_asr", "base_time_s"])
    base["time_s"] = base["time"].apply(parse_time_to_seconds)
    return base[["ratio", "asr", "time_s"]].rename(columns={"asr": "base_asr", "time_s": "base_time_s"})

def _pct_change_df(df: pd.DataFrame, approach: str) -> pd.DataFrame:
    """
    Computes percent change vs baseline ASR for matching ratio.
    Returns columns: approach, tolerance, ratio, asr, time_s, pct_change_asr, dt_s
    pct_change_asr = 100 * (asr / base_asr - 1)
    dt_s = time_s - base_time_s
    """
    sub = df[df["approach"] == approach].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["time_s"] = sub["time"].apply(parse_time_to_seconds)
    base = _baseline_by_ratio(df)
    if base.empty:
        return pd.DataFrame()
    merged = pd.merge(sub, base, on="ratio", how="inner")
    merged["pct_change_asr"] = 100.0 * (merged["asr"] / merged["base_asr"] - 1.0)
    merged["dt_s"] = merged["time_s"] - merged["base_time_s"]
    return merged

def make_pct_change_heatmap(df: pd.DataFrame, filename: str, approach: str):
    """
    Heatmap of percent ASR change vs baseline across (ratio,tolerance) for one approach.
    """
    pct_df = _pct_change_df(df, approach)
    if pct_df.empty:
        print(f"No {approach} data with baseline comparison found in {filename}; skipping pct change heatmap.")
        return
    
    pct_df["tolerance"] = pd.to_numeric(pct_df["tolerance"], errors="coerce")
    pct_df["ratio"] = pd.to_numeric(pct_df["ratio"], errors="coerce")
    pct_df["tol_round"] = pct_df["tolerance"].round(2)
    
    # Create pivot table
    pivot = pct_df.pivot_table(index="tol_round", columns="ratio", values="pct_change_asr", aggfunc="mean")
    
    if pivot.empty:
        print(f"No pivot data for {approach} in {filename}")
        return
    
    # Create the heatmap
    plt.figure(figsize=(12, max(6, len(pivot.index) * 0.6)))
    sns.set_style("white")
    
    # Use a diverging colormap centered at 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmin = -vmax
    
    ax = sns.heatmap(
        pivot,
        cmap="RdBu_r",  # Red=increase (bad), Blue=decrease (good)
        vmin=vmin, vmax=vmax,
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "ASR Change vs Baseline (%)"},
        linewidths=0.5,
        linecolor='white'
    )
    
    # Extract the number from filename for the title
    match = re.search(r'(\d+)fulldata\.csv', filename)
    file_num = match.group(1) if match else "Unknown"
    
    plt.title(f"ASR % Change vs Baseline — {file_num}% Matching — {approach.capitalize()}", fontsize=16, pad=20)
    plt.xlabel("Encryption Ratio (%)", fontsize=12)
    plt.ylabel("Tolerance", fontsize=12)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # Save with dataset number and approach in filename
    base_name = os.path.splitext(filename)[0]  # Remove .csv
    out = os.path.join(OUT_DIR, f"{base_name}_pct_change_heatmap_{approach}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def process_csv_file(csv_path: str):
    """
    Process a single CSV file and generate all heatmaps for it.
    """
    filename = os.path.basename(csv_path)
    print(f"\n=== Processing {filename} ===")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure numeric types
        df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
        df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
        df["asr"] = pd.to_numeric(df["asr"], errors="coerce")
        
        print(f"Loaded {len(df)} rows from {filename}")
        print(f"Approaches found: {sorted(df['approach'].unique())}")
        print(f"Tolerances found: {sorted(df['tolerance'].unique())}")
        print(f"Ratios found: {sorted(df['ratio'].unique())}")
        
        # Generate all heatmaps
        make_raw_asr_heatmap(df, filename)
        
        # Generate separate heatmaps for each non-baseline approach
        approaches = [app for app in df['approach'].unique() if app != 'none']
        for approach in approaches:
            make_raw_asr_heatmap_separate(df, filename, approach)
            make_pct_change_heatmap(df, filename, approach)
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main():
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Find all CSV files matching the pattern [NUM]fulldata.csv
    pattern = "*fulldata.csv"
    csv_files = glob.glob(pattern)
    
    # Filter to only files that match the numeric pattern
    numeric_files = []
    for file in csv_files:
        if re.match(r'^\d+fulldata\.csv$', os.path.basename(file)):
            numeric_files.append(file)
    
    if not numeric_files:
        print("No CSV files found matching pattern [NUM]fulldata.csv")
        return
    
    # Sort by the numeric part
    numeric_files.sort(key=lambda x: int(re.search(r'^(\d+)fulldata\.csv$', os.path.basename(x)).group(1)))
    
    print(f"Found {len(numeric_files)} CSV files matching pattern:")
    for file in numeric_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    for csv_file in numeric_files:
        process_csv_file(csv_file)
    
    print(f"\n✓ All heatmaps saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()