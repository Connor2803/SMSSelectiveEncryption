import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

IN_CSV = "NewResults.csv"
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

def sec_formatter():
    return FuncFormatter(lambda s, _: f"{int(s//60)}m{(s%60):.0f}s")

def pct_formatter():
    return FuncFormatter(lambda y, _: f"{y:.0f}%")

def make_baseline_plot_flipped(df: pd.DataFrame):
    base = df[df["approach"] == "none"].copy()
    if base.empty:
        print("No baseline rows (approach=='none') found; skipping baseline plot.")
        return
    base["time_s"] = base["time"].apply(parse_time_to_seconds)
    base = base.sort_values("asr")  # sort along x (ASR)

    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    plt.plot(base["asr"], base["time_s"], "o-", label="Baseline SE")
    for _, r in base.iterrows():
        plt.annotate(f"r={int(r['ratio'])}%", (r["asr"], r["time_s"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("ASR")
    plt.ylabel("Runtime (seconds)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(sec_formatter())
    plt.title("Baseline SE — Time vs ASR")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "baseline_none_time_vs_asr.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")

def make_baseline_ratio_vs_asr(df: pd.DataFrame):
    base = df[df["approach"] == "none"].copy()
    if base.empty:
        print("No baseline rows (approach=='none') found; skipping baseline ratio plot.")
        return
    base["ratio"] = pd.to_numeric(base["ratio"], errors="coerce")
    base["asr"] = pd.to_numeric(base["asr"], errors="coerce")
    base = base.sort_values("ratio")

    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    plt.plot(base["ratio"], base["asr"], "o-", label="Baseline SE")
    plt.xlabel("Encryption ratio (%)")
    plt.ylabel("ASR")
    plt.title("Baseline SE — ASR vs Encryption Ratio")
    plt.xlim(base["ratio"].min(), base["ratio"].max())
    plt.ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "baseline_none_ratio_vs_asr.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")

def make_individual_tolerance_plots(df: pd.DataFrame, approach: str):
    """
    Create individual plots for each tolerance value, with ratio on x-axis and ASR on y-axis.
    """
    method_data = df[df["approach"] == approach].copy()
    if method_data.empty:
        print(f"No {approach} rows found; skipping individual tolerance plots.")
        return
    
    method_data["ratio"] = pd.to_numeric(method_data["ratio"], errors="coerce")
    method_data["asr"] = pd.to_numeric(method_data["asr"], errors="coerce")
    method_data["tolerance"] = pd.to_numeric(method_data["tolerance"], errors="coerce")
    method_data["tol_round"] = method_data["tolerance"].round(2)
    
    # Get unique tolerances
    tolerances = sorted(method_data["tol_round"].unique())
    
    for tol in tolerances:
        tol_data = method_data[method_data["tol_round"] == tol].copy()
        tol_data = tol_data.sort_values("ratio")
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        plt.plot(tol_data["ratio"], tol_data["asr"], "o-", linewidth=2, markersize=8, 
                color="darkblue" if approach == "sliding" else "darkred")
        
        # Add value annotations
        for _, row in tol_data.iterrows():
            plt.annotate(f"{row['asr']:.3f}", 
                        (row["ratio"], row["asr"]),
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=9, alpha=0.8)
        
        plt.xlabel("Encryption Ratio (%)", fontsize=12)
        plt.ylabel("Attack Success Rate (ASR)", fontsize=12)
        plt.title(f"{approach.capitalize()} Zero-Sum Fuzzing — Tolerance {tol}", fontsize=14)
        plt.xlim(tol_data["ratio"].min()-2, tol_data["ratio"].max()+2)
        plt.ylim(0, max(1, tol_data["asr"].max() * 1.1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with tolerance in filename
        tol_str = f"{tol:.1f}".replace(".", "_")
        out = os.path.join(OUT_DIR, f"{approach}_tolerance_{tol_str}_ratio_vs_asr.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def make_individual_tolerance_time_plots(df: pd.DataFrame, approach: str):
    """
    Create individual plots for each tolerance value, with ratio on x-axis and runtime on y-axis.
    """
    method_data = df[df["approach"] == approach].copy()
    if method_data.empty:
        print(f"No {approach} rows found; skipping individual tolerance time plots.")
        return
    
    method_data["ratio"] = pd.to_numeric(method_data["ratio"], errors="coerce")
    method_data["time_s"] = method_data["time"].apply(parse_time_to_seconds)
    method_data["tolerance"] = pd.to_numeric(method_data["tolerance"], errors="coerce")
    method_data["tol_round"] = method_data["tolerance"].round(2)
    
    # Get unique tolerances
    tolerances = sorted(method_data["tol_round"].unique())
    
    for tol in tolerances:
        tol_data = method_data[method_data["tol_round"] == tol].copy()
        tol_data = tol_data.sort_values("ratio")
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        plt.plot(tol_data["ratio"], tol_data["time_s"], "o-", linewidth=2, markersize=8,
                color="darkgreen" if approach == "sliding" else "darkorange")
        
        # Add time annotations
        for _, row in tol_data.iterrows():
            time_label = f"{int(row['time_s']//60)}m{(row['time_s']%60):.0f}s"
            plt.annotate(time_label, 
                        (row["ratio"], row["time_s"]),
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=9, alpha=0.8)
        
        plt.xlabel("Encryption Ratio (%)", fontsize=12)
        plt.ylabel("Runtime (seconds)", fontsize=12)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(sec_formatter())
        plt.title(f"{approach.capitalize()} Zero-Sum Fuzzing — Runtime, Tolerance {tol}", fontsize=14)
        plt.xlim(tol_data["ratio"].min()-2, tol_data["ratio"].max()+2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with tolerance in filename
        tol_str = f"{tol:.1f}".replace(".", "_")
        out = os.path.join(OUT_DIR, f"{approach}_tolerance_{tol_str}_ratio_vs_time.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def make_combined_tolerance_comparison(df: pd.DataFrame):
    """
    Create a combined plot showing both methods across all tolerances, with ratio on x-axis.
    """
    sliding_data = df[df["approach"] == "sliding"].copy()
    adaptive_data = df[df["approach"] == "adaptive"].copy()
    
    if sliding_data.empty and adaptive_data.empty:
        print("No sliding or adaptive data found; skipping combined comparison.")
        return
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Prepare data
    for data, approach in [(sliding_data, "sliding"), (adaptive_data, "adaptive")]:
        if data.empty:
            continue
        data["ratio"] = pd.to_numeric(data["ratio"], errors="coerce")
        data["asr"] = pd.to_numeric(data["asr"], errors="coerce")
        data["tolerance"] = pd.to_numeric(data["tolerance"], errors="coerce")
        data["tol_round"] = data["tolerance"].round(2)
        
        # Plot with different colors/markers for each tolerance
        tolerances = sorted(data["tol_round"].unique())
        colors = plt.cm.Set1(range(len(tolerances)))
        
        for i, tol in enumerate(tolerances):
            tol_data = data[data["tol_round"] == tol].sort_values("ratio")
            marker = "o" if approach == "sliding" else "s"
            linestyle = "-" if approach == "sliding" else "--"
            
            plt.plot(tol_data["ratio"], tol_data["asr"], 
                    marker=marker, linestyle=linestyle, linewidth=2, markersize=6,
                    color=colors[i], 
                    label=f"{approach.capitalize()} (tol={tol})")
    
    plt.xlabel("Encryption Ratio (%)", fontsize=12)
    plt.ylabel("Attack Success Rate (ASR)", fontsize=12)
    plt.title("Zero-Sum Fuzzing Methods Comparison — ASR vs Encryption Ratio", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out = os.path.join(OUT_DIR, "combined_methods_all_tolerances_ratio_vs_asr.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def make_raw_asr_heatmaps(df: pd.DataFrame):
    """
    Create heatmaps showing raw ASR values for all methods, with baseline at the top.
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
    pivot = pivot.reindex(ordered_index)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.set_style("white")
    
    # Use a color map that highlights low values (better security)
    ax = sns.heatmap(
        pivot,
        cmap="RdYlBu_r",  # Red for high ASR (bad), Blue for low ASR (good)
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Attack Success Rate (ASR)"},
        linewidths=0.5,
        linecolor='white'
    )
    
    plt.title("Raw ASR Values — All Methods and Tolerances", fontsize=16, pad=20)
    plt.xlabel("Encryption Ratio (%)", fontsize=12)
    plt.ylabel("Method (Tolerance)", fontsize=12)
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    out = os.path.join(OUT_DIR, "raw_asr_heatmap_all_methods.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def make_raw_asr_heatmaps_separate(df: pd.DataFrame):
    """
    Create separate heatmaps for sliding and adaptive methods, each with baseline comparison.
    """
    for approach in ["sliding", "adaptive"]:
        # Get data for this approach and baseline
        method_data = df[df["approach"].isin([approach, "none"])].copy()
        if method_data.empty:
            continue
            
        method_data["ratio"] = pd.to_numeric(method_data["ratio"], errors="coerce")
        method_data["asr"] = pd.to_numeric(method_data["asr"], errors="coerce")
        method_data["tolerance"] = pd.to_numeric(method_data["tolerance"], errors="coerce")
        
        # Create method-tolerance labels
        method_data["method_tol"] = method_data.apply(lambda row: 
            f"Baseline (SE only)" if row["approach"] == "none" 
            else f"{row['approach'].capitalize()} (tol={row['tolerance']:.2f})", axis=1)
        
        # Create pivot table
        pivot = method_data.pivot_table(index="method_tol", columns="ratio", values="asr", aggfunc="mean")
        
        # Sort to put baseline at top
        baseline_rows = [idx for idx in pivot.index if "Baseline" in idx]
        method_rows = sorted([idx for idx in pivot.index if approach.capitalize() in idx])
        ordered_index = baseline_rows + method_rows
        pivot = pivot.reindex(ordered_index)
        
        # Create the heatmap
        plt.figure(figsize=(10, 6))
        sns.set_style("white")
        
        ax = sns.heatmap(
            pivot,
            cmap="RdYlBu_r",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Attack Success Rate (ASR)"},
            linewidths=0.5,
            linecolor='white'
        )
        
        plt.title(f"Raw ASR Values — {approach.capitalize()} Zero-Sum Fuzzing vs Baseline", 
                 fontsize=14, pad=20)
        plt.xlabel("Encryption Ratio (%)", fontsize=12)
        plt.ylabel("Method (Tolerance)", fontsize=12)
        
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        out = os.path.join(OUT_DIR, f"raw_asr_heatmap_{approach}_vs_baseline.png")
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
        return sub
    sub["time_s"] = sub["time"].apply(parse_time_to_seconds)
    base = _baseline_by_ratio(df)
    merged = pd.merge(sub, base, on="ratio", how="inner")
    merged["pct_change_asr"] = 100.0 * (merged["asr"] / merged["base_asr"] - 1.0)
    merged["dt_s"] = merged["time_s"] - merged["base_time_s"]
    return merged

def make_individual_tolerance_improvement_plots(df: pd.DataFrame):
    """
    For each tolerance, create a plot showing ASR improvement vs baseline across encryption ratios.
    """
    base = _baseline_by_ratio(df)
    if base.empty:
        print("No baseline rows found; skipping improvement plots.")
        return

    sliding_df = _pct_change_df(df, "sliding")
    adaptive_df = _pct_change_df(df, "adaptive")

    if sliding_df.empty and adaptive_df.empty:
        print("No sliding/adaptive rows found; skipping improvement plots.")
        return

    # Get all tolerances
    all_tolerances = set()
    if not sliding_df.empty:
        all_tolerances.update(sliding_df["tolerance"].round(2).unique())
    if not adaptive_df.empty:
        all_tolerances.update(adaptive_df["tolerance"].round(2).unique())
    
    all_tolerances = sorted(all_tolerances)

    for tol in all_tolerances:
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        # Zero reference line
        plt.axhline(0, color="gray", linewidth=1, linestyle="--", zorder=0, alpha=0.7)
        
        # Plot sliding if available
        if not sliding_df.empty:
            tol_sliding = sliding_df[sliding_df["tolerance"].round(2) == tol].sort_values("ratio")
            if not tol_sliding.empty:
                plt.plot(tol_sliding["ratio"], tol_sliding["pct_change_asr"], 
                        "o-", linewidth=3, markersize=8, color="darkblue", 
                        label="Sliding Window Fuzzing")
                
                # Annotate with actual ASR values
                for _, row in tol_sliding.iterrows():
                    plt.annotate(f"ASR: {row['asr']:.3f}", 
                                (row["ratio"], row["pct_change_asr"]),
                                textcoords="offset points", xytext=(5, 5), 
                                fontsize=9, alpha=0.8)

        # Plot adaptive if available
        if not adaptive_df.empty:
            tol_adaptive = adaptive_df[adaptive_df["tolerance"].round(2) == tol].sort_values("ratio")
            if not tol_adaptive.empty:
                plt.plot(tol_adaptive["ratio"], tol_adaptive["pct_change_asr"], 
                        "s-", linewidth=3, markersize=8, color="darkred", 
                        label="Adaptive Pattern Fuzzing")
                
                # Annotate with actual ASR values
                for _, row in tol_adaptive.iterrows():
                    plt.annotate(f"ASR: {row['asr']:.3f}", 
                                (row["ratio"], row["pct_change_asr"]),
                                textcoords="offset points", xytext=(5, -15), 
                                fontsize=9, alpha=0.8)

        plt.xlabel("Encryption Ratio (%)", fontsize=12)
        plt.ylabel("ASR Improvement vs Baseline (%)", fontsize=12)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(pct_formatter())
        plt.title(f"Zero-Sum Fuzzing Performance — Tolerance {tol:.1f}", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with tolerance in filename
        tol_str = f"{tol:.1f}".replace(".", "_")
        out = os.path.join(OUT_DIR, f"improvement_tolerance_{tol_str}_ratio_vs_pct_change.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

def make_pct_change_heatmaps(df: pd.DataFrame):
    """
    Heatmaps of percent ASR change vs baseline across (ratio,tolerance) for each approach.
    """
    for approach in ["sliding", "adaptive"]:
        m = _pct_change_df(df, approach)
        if m.empty:
            print(f"No rows for {approach}; skipping heatmap.")
            continue
        # Pivot: rows=tolerance, cols=ratio
        pivot = m.pivot_table(index="tolerance", columns="ratio", values="pct_change_asr", aggfunc="mean")
        pivot = pivot.sort_index()  # sort by tolerance

        plt.figure(figsize=(10, 6))
        sns.set_style("white")
        ax = sns.heatmap(
            pivot,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "ASR change vs baseline (%)"}
        )
        plt.title(f"{approach.capitalize()} Zero-Sum Fuzzing | ASR Improvement (%)", fontsize=14)
        plt.xlabel("Encryption Ratio (%)", fontsize=12)
        plt.ylabel("Tolerance", fontsize=12)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"pct_change_heatmap_{approach}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    
    # Ensure numeric types
    df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce")

    # Original plots
    #make_baseline_ratio_vs_asr(df)
    
    # NEW: Individual plots separated by tolerance
    #print("\n=== Creating individual plots by tolerance ===")
    #make_individual_tolerance_plots(df, "sliding")
    #make_individual_tolerance_plots(df, "adaptive")
    #make_individual_tolerance_time_plots(df, "sliding")
    #make_individual_tolerance_time_plots(df, "adaptive")
    
    # NEW: Combined comparison
    #print("\n=== Creating combined comparison ===")
    make_combined_tolerance_comparison(df)
    
    # NEW: Individual improvement plots by tolerance
    #print("\n=== Creating improvement plots by tolerance ===")
    #make_individual_tolerance_improvement_plots(df)
    
    # NEW: Raw ASR value heatmaps
    print("\n=== Creating raw ASR heatmaps ===")
    make_raw_asr_heatmaps(df)
    make_raw_asr_heatmaps_separate(df)
    
    # Percentage change heatmaps
    print("\n=== Creating percentage change heatmaps ===")
    make_pct_change_heatmaps(df)

    print(f"\n✓ All plots saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()