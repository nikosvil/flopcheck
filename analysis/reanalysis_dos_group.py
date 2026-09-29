"""Re-analysis of a public GPU benchmark through Annex A.2.1 of the GPAI Guidelines.

Source data
    Enskat & Wiesner (2026), "Evaluating MFU as a Proxy for GPU Power for
    Energy-Aware Simulation of LLM Training", arXiv:2608.03880.
    github.com/dos-group/gpu_power_benchmark (MIT, (c) Enskat, Wiesner & Kao); the commit used is
    recorded in data/dos-group-gpu-power-benchmark/SOURCE_COMMIT.txt.

Question
    On the *same* training runs, how far apart are two quantities a provider
    could report, in good faith, as Annex A.2.1's "average percentage of GPU
    utilisation"?  (i) the vendor utilisation counter (nvidia-smi / rocm-smi)
    and (ii) model FLOPs utilisation (MFU). Because Annex A.2.1 is linear in U,
    their ratio is the factor by which the reported training compute changes.

Row filters, and why
    * the idle 'baseline' row is dropped;
    * only the paper's measurement protocol is kept (5 warm-up steps, 5 s
      cool-down);
    * only bf16 and fp16 are kept. For "fp32" the repository normalises by a
      different number-format peak on each device (TF32, FP16 or BF16 tensor
      paths, or the FP32 CUDA-core peak), and on two devices the resulting fp32
      MFU implies throughput about 4x above the device's published FP32 peak --
      see fp32_peak_report();
    * MFU is the repository's profiler-based column (CalFLOPS), which the paper
      uses and which matches 6*P*D for the models checked here. The repository's
      second, formula-based column is compared in figure 3.

Caveat
    Every run uses an 11-token input sequence (sequence_length_mean == 11 at both
    'context window' settings), so steps are very small. Production pre-training
    runs at higher MFU; the ratios here illustrate that the two quantities are
    different and diverge systematically, not their size in a frontier run.

Run:  python analysis/reanalysis_dos_group.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dos-group-gpu-power-benchmark"
OUT = ROOT / "analysis" / "out"

ORDER = ["A100", "L40", "L4", "4070", "GPU06", "MI210"]
GPU_LABEL = {
    "A100": "NVIDIA A100",
    "L40": "NVIDIA L40",
    "L4": "NVIDIA L4",
    "4070": "NVIDIA RTX 4070 Ti",
    "GPU06": "NVIDIA Quadro RTX 5000",
    "MI210": "AMD MI210",
}

# Peaks the repository uses for "fp32" (benchmark.py, pick_peak_tflops), and the
# device's own published FP32 peaks where verified against a vendor datasheet.
REPO_FP32_PEAK_TFLOPS = {"A100": 156, "L40": 90.5, "L4": 30.3, "4070": 40.1, "GPU06": 89.2, "MI210": 181}
DEVICE_FP32_PEAK_TFLOPS = {
    # AMD MI210 brochure: FP32 vector 22.6, FP32 matrix 45.3, FP16/BF16 181.
    "MI210": ("FP32 matrix 45.3 (vector 22.6)", 45.3),
    # NVIDIA Quadro RTX 5000 datasheet: FP32 11.2, tensor 89.2.
    "GPU06": ("FP32 11.2 (tensor 89.2)", 11.2),
}

# Published parameter counts, for checking the two FLOP-counting methods.
PARAMS = {
    "Qwen/Qwen2.5-0.5B": 0.49e9,
    "Qwen/Qwen2.5-1.5B": 1.54e9,
    "Qwen/Qwen2.5-3B": 3.09e9,
    "gpt2": 0.124e9,
    "gpt2-xl": 1.558e9,
    "microsoft/DialoGPT-medium": 0.355e9,
    "microsoft/DialoGPT-large": 0.774e9,
}
SHORT = {k: k.split("/")[-1] for k in PARAMS}

# Reference palette (dataviz skill, light mode), validated for two series.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S_MFU, S_UTIL = "#2a78d6", "#eb6834"  # categorical slots 1 and 2

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
    }
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load() -> pd.DataFrame:
    frames = []
    for g in ORDER:
        d = pd.read_csv(DATA / f"mfu_aggregated_per_config_{g}.csv")
        d["gpu"] = g
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def protocol(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.model_name != "baseline"]
    d = d[(d.warmup_iterations == 5) & (d.cooldown_seconds == 5.0)]
    if "io_latency_enabled" in d:
        d = d[~d.io_latency_enabled.fillna(False).astype(bool)]
    return d


def clean(df: pd.DataFrame) -> pd.DataFrame:
    d = protocol(df)
    d = d[d.dtype.isin(["bfloat16", "float16"]) & (d.mfu_percentage_calflops_mean > 0)]
    d = d.assign(
        mfu=d.mfu_percentage_calflops_mean / 100,
        mfu_formula=d.mfu_percentage_mean / 100,
        util=d.gpu_utilization_mean / 100,
    )
    d["util_over_mfu"] = d.util / d.mfu
    d["calflops_over_formula"] = d.mfu / d.mfu_formula
    return d


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def by_batch(d: pd.DataFrame) -> pd.DataFrame:
    return (
        d.groupby(["gpu", "batch_size"])
        .agg(
            mfu=("mfu", "median"), mfu_lo=("mfu", "min"), mfu_hi=("mfu", "max"),
            util=("util", "median"), util_lo=("util", "min"), util_hi=("util", "max"),
            n=("mfu", "size"),
        )
        .reset_index()
    )


def at_batch(d: pd.DataFrame, batch: int = 128) -> pd.DataFrame:
    b = (
        d[d.batch_size == batch]
        .groupby("gpu")
        .agg(
            mfu=("mfu", "median"), mfu_lo=("mfu", "min"), mfu_hi=("mfu", "max"),
            util=("util", "median"), util_lo=("util", "min"), util_hi=("util", "max"),
            ratio_lo=("util_over_mfu", "min"), ratio_hi=("util_over_mfu", "max"),
            n=("mfu", "size"),
        )
    )
    b["ratio"] = b.util / b.mfu
    return b.reindex([g for g in ORDER if g in b.index])


def flop_methods(d: pd.DataFrame) -> pd.DataFrame:
    big = d[d.batch_size >= 16].copy()
    tokens = big.batch_size * big.sequence_length_mean
    big["calflops_per_token"] = big.calflops_total_mean / tokens
    t = big.groupby("model_name").agg(
        ratio=("calflops_over_formula", "median"),
        calflops_per_token=("calflops_per_token", "median"),
        formula_per_token=("formula_flops_per_token_mean", "median"),
    )
    t["six_n"] = [6 * PARAMS[m] for m in t.index]
    t["calflops_over_6n"] = t.calflops_per_token / t.six_n
    t["formula_over_6n"] = t.formula_per_token / t.six_n
    return t.sort_values("ratio")


def fp32_peak_report(df: pd.DataFrame) -> pd.DataFrame:
    f = protocol(df)
    f = f[f.dtype == "float32"]
    t = f.groupby("gpu").agg(max_mfu=("mfu_percentage_calflops_mean", "max"))
    t["repo_fp32_peak_tflops"] = [REPO_FP32_PEAK_TFLOPS[g] for g in t.index]
    t["implied_tflops"] = t.max_mfu / 100 * t.repo_fp32_peak_tflops
    t["device_fp32_peak"] = [DEVICE_FP32_PEAK_TFLOPS.get(g, ("not checked", None))[0] for g in t.index]
    t["implied_over_device_peak"] = [
        (t.implied_tflops[g] / DEVICE_FP32_PEAK_TFLOPS[g][1]) if g in DEVICE_FP32_PEAK_TFLOPS else None
        for g in t.index
    ]
    return t.reindex([g for g in ORDER if g in t.index])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=0, labelcolor=INK2, labelsize=8.5)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def _title(fig, title, subtitle, y=0.975):
    fig.text(0.02, y, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.02, y - 0.055, subtitle, ha="left", va="top", fontsize=9, color=INK2)


SOURCE = "Data: Enskat & Wiesner (2026), arXiv:2608.03880; github.com/dos-group/gpu_power_benchmark (MIT). Re-analysis: flopcheck."


def fig_small_multiples(bb: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.77, bottom=0.12, hspace=0.36, wspace=0.08)
    for ax, g in zip(axes.flat, ORDER):
        s = bb[bb.gpu == g]
        x = s.batch_size
        ax.fill_between(x, s.util_lo * 100, s.util_hi * 100, color=S_UTIL, alpha=0.10, lw=0)
        ax.fill_between(x, s.mfu_lo * 100, s.mfu_hi * 100, color=S_MFU, alpha=0.10, lw=0)
        for col, colour in (("util", S_UTIL), ("mfu", S_MFU)):
            ax.plot(x, s[col] * 100, color=colour, lw=1.6, marker="o", ms=5.5,
                    mec=SURFACE, mew=1.2, solid_capstyle="round", solid_joinstyle="round")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 128], ["1", "4", "16", "64", "128"])
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(GPU_LABEL[g], loc="left", fontsize=10, fontweight="semibold")
        _style(ax)
        if g == "MI210":
            ax.text(1, 88, "counter is a binary activity flag", fontsize=8, color=MUTED, va="top")
    handles = [
        Line2D([], [], color=S_UTIL, lw=1.6, marker="o", ms=5.5, mec=SURFACE, label="GPU utilization counter (nvidia-smi / rocm-smi)"),
        Line2D([], [], color=S_MFU, lw=1.6, marker="o", ms=5.5, mec=SURFACE, label="Model FLOP utilization (MFU)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.868), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, 'Same training runs, two readings of "GPU utilization"',
           "Median across models, bf16 and fp16; shaded band = range across models and precisions.")
    fig.supxlabel("Batch size (sequences of 11 tokens)", fontsize=9, color=INK2, y=0.045)
    fig.supylabel("Percent", fontsize=9, color=INK2, x=0.012)
    fig.text(0.02, 0.008, SOURCE, fontsize=7.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def fig_dumbbell(b: pd.DataFrame, path: Path) -> None:
    rows = b.sort_values("ratio").index.tolist()
    nv = b.drop(index="MI210", errors="ignore")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.subplots_adjust(left=0.27, right=0.9, top=0.72, bottom=0.14)
    for i, g in enumerate(rows):
        m, u, r = b.mfu[g] * 100, b.util[g] * 100, b.ratio[g]
        ax.plot([m, u], [i, i], color=AXIS, lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter([m], [i], s=64, color=S_MFU, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([u], [i], s=64, color=S_UTIL, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(104, i, f"×{r:.1f}", va="center", ha="left", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(rows)),
                  [GPU_LABEL[g] + ("  (binary counter)" if g == "MI210" else "") for g in rows])
    ax.set_xlim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Percent", fontsize=9)
    _style(ax, grid_axis="x")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    handles = [
        Line2D([], [], ls="", marker="o", ms=7, color=S_MFU, mec=SURFACE, label="MFU"),
        Line2D([], [], ls="", marker="o", ms=7, color=S_UTIL, mec=SURFACE, label="GPU utilization counter"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.83), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig,
           f"At batch 128, the median counter reading is at least {b.ratio.min():.0f}× the MFU on every GPU tested",
           "Annex A.2.1 is linear in U, so each ratio is the factor by which reported training compute changes\n"
           "with the reading of \"GPU utilization\". Medians over models, bf16 and fp16; 11-token sequences.")
    fig.text(0.02, 0.012, SOURCE, fontsize=7.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def fig_flop_methods(t: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    fig.subplots_adjust(left=0.24, right=0.92, top=0.76, bottom=0.14)
    for i, (name, row) in enumerate(t.iterrows()):
        v = row.ratio
        ax.plot([1, v], [i, i], color=S_MFU, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([v], [i], s=64, color=S_MFU, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(v + 0.04, i, f"{v:.2f}×", va="center", ha="left", fontsize=9.5, color=INK2)
    ax.axvline(1, color=AXIS, lw=0.8, zorder=1)
    ax.set_yticks(range(len(t)), [SHORT[m] for m in t.index])
    ax.set_xlim(0.9, t.ratio.max() + 0.35)
    ax.set_xlabel("Profiler-based MFU ÷ formula-based MFU (1 = agreement)", fontsize=9)
    _style(ax, grid_axis="x")
    _title(fig,
           f"Two FLOP counts on the same runs disagree by up to {t.ratio.max():.1f}×",
           "The repository records MFU from a profiler (CalFLOPS) and from an analytic formula.\n"
           "Median over bf16 and fp16 runs at batch ≥ 16. The profiler column matches 6·P·D.")
    fig.text(0.02, 0.012, SOURCE, fontsize=7.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load()
    d = clean(raw)

    seq = sorted(d.sequence_length_mean.unique().tolist())
    bb, b128, fm, fp = by_batch(d), at_batch(d), flop_methods(d), fp32_peak_report(raw)

    bb.to_csv(OUT / "fig1_by_batch.csv", index=False)
    b128.to_csv(OUT / "fig2_batch128.csv")
    fm.to_csv(OUT / "fig3_flop_methods.csv")
    fp.to_csv(OUT / "fp32_peak_report.csv")

    fig_small_multiples(bb, OUT / "fig1_util_vs_mfu")
    fig_dumbbell(b128, OUT / "fig2_gap_at_batch128")
    fig_flop_methods(fm, OUT / "fig3_flop_count_methods")

    nv = b128.drop(index="MI210", errors="ignore")
    summary = {
        "source_commit": (DATA / "SOURCE_COMMIT.txt").read_text().strip(),
        "rows_used": int(len(d)),
        "sequence_lengths_in_data": seq,
        "batch128_nvidia_ratio_median_range": [round(nv.ratio.min(), 2), round(nv.ratio.max(), 2)],
        "batch128_nvidia_ratio_all_rows_range": [round(nv.ratio_lo.min(), 2), round(nv.ratio_hi.max(), 2)],
        "batch128_mfu_median_range_pct": [round(nv.mfu.min() * 100, 1), round(nv.mfu.max() * 100, 1)],
        "batch128_util_median_range_pct": [round(nv.util.min() * 100, 1), round(nv.util.max() * 100, 1)],
        "mi210_util_min_pct": round(float(d[d.gpu == "MI210"].util.min() * 100), 1),
        "flop_method_ratio_range": [round(fm.ratio.min(), 2), round(fm.ratio.max(), 2)],
        "calflops_over_6n_range": [round(fm.calflops_over_6n.min(), 2), round(fm.calflops_over_6n.max(), 2)],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    print("\nBatch 128:\n", b128.round(3).to_string())
    print("\nFLOP-count methods:\n", fm.round(3).to_string())
    print("\nfp32 peak report:\n", fp.round(2).to_string())


if __name__ == "__main__":
    main()
