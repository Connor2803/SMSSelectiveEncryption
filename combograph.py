import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

IN_CSV = "results.csv"
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

if __name__ == "__main__":
    main()