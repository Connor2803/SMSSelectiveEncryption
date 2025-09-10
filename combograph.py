import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

IN_CSV = "results8.csv"
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

def make_sliding_combined_flipped(df: pd.DataFrame):
    sl = df[df["approach"] == "sliding"].copy()
    if sl.empty:
        print("No sliding rows found; skipping sliding plot.")
        return
    sl["time_s"] = sl["time"].apply(parse_time_to_seconds)
    sl["tol_round"] = sl["tolerance"].round(2)

    # Ensure lines are connected in a consistent order per tolerance
    sl = sl.sort_values(["tol_round", "ratio", "asr"])

    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=sl, x="asr", y="time_s", hue="tol_round", marker="o", palette="tab10")
    plt.xlabel("ASR")
    plt.ylabel("Runtime (seconds)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(sec_formatter())
    plt.title("Sliding — Time vs ASR for all tolerances")
    plt.legend(title="Tolerance", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sliding_all_time_vs_asr.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
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

def make_sliding_combined_ratio_vs_asr(df: pd.DataFrame):
    sl = df[df["approach"] == "sliding"].copy()
    if sl.empty:
        print("No sliding rows found; skipping sliding ratio plot.")
        return
    sl["ratio"] = pd.to_numeric(sl["ratio"], errors="coerce")
    sl["asr"] = pd.to_numeric(sl["asr"], errors="coerce")
    sl["tol_round"] = pd.to_numeric(sl["tolerance"], errors="coerce").round(2)
    sl = sl.sort_values(["tol_round", "ratio"])

    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=sl, x="ratio", y="asr", hue="tol_round", marker="o", palette="tab10")
    plt.xlabel("Encryption ratio (%)")
    plt.ylabel("ASR")
    plt.title("Sliding — ASR vs Encryption Ratio (all tolerances)")
    plt.xlim(sl["ratio"].min(), sl["ratio"].max())
    plt.ylim(0, 1)
    plt.legend(title="Tolerance", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sliding_all_ratio_vs_asr.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def make_adaptive_combined_flipped(df: pd.DataFrame):
    ad = df[df["approach"] == "adaptive"].copy()
    if ad.empty:
        print("No adaptive rows found; skipping adaptive plot.")
        return
    ad["time_s"] = ad["time"].apply(parse_time_to_seconds)
    ad["tol_round"] = ad["tolerance"].round(2)
    ad = ad.sort_values(["tol_round", "ratio", "asr"])

    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=ad, x="asr", y="time_s", hue="tol_round", marker="o", palette="tab10")
    plt.xlabel("ASR")
    plt.ylabel("Runtime (seconds)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(sec_formatter())
    plt.title("Adaptive — Time vs ASR for all tolerances")
    plt.legend(title="Tolerance", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "adaptive_all_time_vs_asr.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def make_adaptive_combined_ratio_vs_asr(df: pd.DataFrame):
    ad = df[df["approach"] == "adaptive"].copy()
    if ad.empty:
        print("No adaptive rows found; skipping adaptive ratio plot.")
        return
    ad["ratio"] = pd.to_numeric(ad["ratio"], errors="coerce")
    ad["asr"] = pd.to_numeric(ad["asr"], errors="coerce")
    ad["tol_round"] = pd.to_numeric(ad["tolerance"], errors="coerce").round(2)
    ad = ad.sort_values(["tol_round", "ratio"])

    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=ad, x="ratio", y="asr", hue="tol_round", marker="o", palette="tab10")
    plt.xlabel("Encryption ratio (%)")
    plt.ylabel("ASR")
    plt.title("Adaptive — ASR vs Encryption Ratio (all tolerances)")
    plt.xlim(ad["ratio"].min(), ad["ratio"].max())
    plt.ylim(0, 1)
    plt.legend(title="Tolerance", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "adaptive_all_ratio_vs_asr.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

def pct_formatter():
    return FuncFormatter(lambda y, _: f"{y:.0f}%")

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

def make_pct_change_plots(df: pd.DataFrame):
    """
    For each ratio available in the baseline, plot percent ASR change vs tolerance
    for both Sliding and Adaptive compared to baseline at that ratio.
    Saves plots/pct_change_asr_vs_tol_ratio_{ratio}.png
    """
    base = _baseline_by_ratio(df)
    if base.empty:
        print("No baseline rows found; skipping percent change plots.")
        return

    # Prepare approach-specific frames
    sliding_df = _pct_change_df(df, "sliding")
    adaptive_df = _pct_change_df(df, "adaptive")

    # If neither has data, nothing to do
    if sliding_df.empty and adaptive_df.empty:
        print("No sliding/adaptive rows found; skipping percent change plots.")
        return

    # For each ratio in baseline, make a combined plot if any approach has data
    for r in sorted(base["ratio"].unique()):
        has_sl = not sliding_df[sliding_df["ratio"] == r].empty
        has_ad = not adaptive_df[adaptive_df["ratio"] == r].empty
        if not (has_sl or has_ad):
            continue

        plt.figure(figsize=(9, 6))
        sns.set_style("whitegrid")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(pct_formatter())

        # Zero reference line
        plt.axhline(0, color="gray", linewidth=1, linestyle="--", zorder=0)

        # Plot each approach (if available)
        if has_sl:
            g = sliding_df[sliding_df["ratio"] == r].sort_values("tolerance")
            plt.plot(g["tolerance"], g["pct_change_asr"], "o-", label="Sliding")
            # Annotate with time delta vs baseline
            for _, row in g.iterrows():
                plt.annotate(f"{row['dt_s']:+.0f}s",
                             (row["tolerance"], row["pct_change_asr"]),
                             textcoords="offset points", xytext=(4, 4), fontsize=8)

        if has_ad:
            g = adaptive_df[adaptive_df["ratio"] == r].sort_values("tolerance")
            plt.plot(g["tolerance"], g["pct_change_asr"], "o-", label="Adaptive")
            for _, row in g.iterrows():
                plt.annotate(f"{row['dt_s']:+.0f}s",
                             (row["tolerance"], row["pct_change_asr"]),
                             textcoords="offset points", xytext=(4, 4), fontsize=8)

        plt.xlabel("Tolerance")
        plt.ylabel("ASR change vs baseline (%)")
        plt.title(f"Percent change in ASR vs baseline (ratio={int(r)}%)")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"pct_change_asr_vs_tol_ratio_{int(r)}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")

def make_pct_change_heatmaps(df: pd.DataFrame):
    """
    Heatmaps of percent ASR change vs baseline across (ratio,tolerance) for each approach.
    Adds a rightmost 'Total' column that sums across all ratios for each tolerance.
    """
    for approach in ["sliding", "adaptive"]:
        m = _pct_change_df(df, approach)
        if m.empty:
            print(f"No rows for {approach}; skipping heatmap.")
            continue
        # Pivot: rows=tolerance, cols=ratio
        pivot = m.pivot_table(index="tolerance", columns="ratio", values="pct_change_asr", aggfunc="mean")
        pivot = pivot.sort_index()  # sort by tolerance
        # Add total across ratios on the far right
        pivot_with_total = pivot.copy()
        pivot_with_total["Total"] = pivot.sum(axis=1)

        plt.figure(figsize=(10, 6))
        sns.set_style("white")
        ax = sns.heatmap(
            pivot_with_total,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "ASR change vs baseline (%)"}
        )
        plt.title(f"{approach.capitalize()} — ASR change vs baseline (%)")
        plt.xlabel("Encryption ratio (%)")
        plt.ylabel("Tolerance")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"pct_change_heatmap_{approach}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Wrote {out}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    for col in ["tolerance", "ratio, asr".split(", ")]:
        pass  # placeholder to avoid accidental edits
    # Ensure numeric types
    df["tolerance"] = pd.to_numeric(df["tolerance"], errors="coerce")
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df["asr"] = pd.to_numeric(df["asr"], errors="coerce")

    make_baseline_plot_flipped(df)
    make_sliding_combined_flipped(df)
    make_adaptive_combined_flipped(df)
    make_baseline_ratio_vs_asr(df)
    make_sliding_combined_ratio_vs_asr(df)
    make_adaptive_combined_ratio_vs_asr(df)

    # New visuals: percent change vs baseline
    make_pct_change_plots(df)
    make_pct_change_heatmaps(df)

if __name__ == "__main__":
    main()