"""Analysis of the H100 probe run through Annex A.1 and A.2.1 of the GPAI Guidelines.

Source
    probe/results/20260916T190612Z/ -- one NVIDIA H100 PCIe (Scaleway H100-1-80G),
    16 September 2026. TinyLlama 1.1B architecture, random tokens, bf16 autocast,
    torch.compile, six configurations of sequence length, batch size and full
    activation checkpointing; each measured for 120 s after a 45 s warm-up.
    nvidia_smi_performance_under_load.txt confirms the GPU was held by its
    software power cap, with no thermal slowdown.

Readings on the same training steps
    counter  NVML utilisation.gpu, the number nvidia-smi shows
    SMACT    DCGM streaming-multiprocessor activity
    TENSO    DCGM tensor-pipe activity
    HFU      FLOPs executed per token (torch flop counter, includes recomputation)
    MFU      model FLOPs per token (6*P + 12*layers*heads*head_dim*seq;
             P is the parameter count of Annex A.2.2; recomputation excluded,
             the Annex A.1 basis)

Peak performance
    MFU and HFU use 756.45 TFLOP/s dense BF16 for the H100 PCIe: 114 streaming
    multiprocessors (NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.01)
    x 4 096 FLOPs per cycle (Pedersen et al. 2026) x a 1 620 MHz tensor-core
    boost clock (NVIDIA developer forum) -- the same derivation that gives
    989.4 for the SXM (132 x 4 096 x 1 830 MHz), the figure the Commission uses
    for Model H. The whitepaper rounds these to 800 and 1 000. Every ratio except
    counter/MFU is independent of the peak; its sensitivity is reported.

OFU
    Tensor-pipe activity x SM clock / 1 620 MHz, following Pedersen et al.'s
    definition with the PCIe tensor-core clock in place of the SXM's 1 830 MHz.

Run:  python analysis/h100_probe_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reanalysis_dos_group import AXIS, INK2, MUTED, S_MFU, S_UTIL, SURFACE, _style, _title  # noqa: E402

ROOT = HERE.parent
RUN = ROOT / "probe" / "results" / "20260916T190612Z"
OUT = HERE / "out"

S_HFU = "#1baf7a"  # categorical slot 3; validated with slots 1-2 for all pairs

PEAK_USED_IN_RUN = 756e12
#: 114 SMs x 4 096 FLOPs/cycle x 1 620 MHz: H100 PCIe, dense BF16 tensor-core peak.
PEAK_PCIE_BF16_DENSE = 114 * 4096 * 1620e6
SCALE = PEAK_USED_IN_RUN / PEAK_PCIE_BF16_DENSE
#: H100 PCIe tensor-core boost clock, for OFU.
TC_MAX_MHZ_PCIE = 1620.0
#: Dense BF16 figures a provider could cite for "an H100".
CANDIDATE_PEAKS_TFLOPS = {"756.45 (PCIe, clock-derived)": 756.45, "800 (PCIe, whitepaper, rounded)": 800.0,
                          "989.4 (SXM, datasheet)": 989.4, "1000 (SXM5, whitepaper, rounded)": 1000.0}
SOURCE = ("Measured 16 Sep 2026 on one NVIDIA H100 PCIe (Scaleway H100-1-80G). TinyLlama-1.1B architecture, "
          "random tokens, bf16, torch.compile. Peak 756.45 TFLOP/s dense BF16.")


def label(c: dict) -> str:
    s = f"{c['seq'] // 1024}k tokens, batch {c['batch']}"
    return s + (", checkpointing" if c["ckpt"] else "")


def load() -> tuple[pd.DataFrame, dict]:
    env = json.loads((RUN / "env.json").read_text())
    rows = []
    for line in (RUN / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        c = r["config"]
        clk = r["dcgm_sm_clock_mhz"]["mean"]
        rows.append({
            "config": label(c), "seq": c["seq"], "batch": c["batch"], "ckpt": c["ckpt"],
            "tokens_per_s": r["tokens_per_s"],
            "mfu": r["mfu"] * SCALE, "mfu_causal_half": r["mfu_causal_half"] * SCALE, "hfu": r["hfu"] * SCALE,
            "counter": r["nvml_util_pct"]["mean"] / 100,
            "smact": r["dcgm_sm_active"]["mean"], "tensor": r["dcgm_tensor_active"]["mean"],
            "sm_clock_mhz": clk, "clock_over_max": clk / env["max_sm_clock_mhz"],
            "power_w": r["nvml_power_w"]["mean"], "energy_j": r["energy_j"],
            "j_per_model_flop": r["joules_per_model_flop"],
            "executed_over_model": r["executed_flops_per_token"] / r["model_flops_per_token"],
            "ofu": r["dcgm_tensor_active"]["mean"] * clk / TC_MAX_MHZ_PCIE,
        })
    d = pd.DataFrame(rows)
    d["counter_over_mfu"] = d.counter / d.mfu
    d["hfu_over_mfu"] = d.hfu / d.mfu
    d["full_over_half"] = d.mfu / d.mfu_causal_half
    return d, env


def _dumbbell(d, left, right, left_colour, right_colour, left_name, right_name, ratio, path,
              title, subtitle, xmax=102, ratio_fmt="×{:.2f}"):
    rows = d.sort_values(ratio).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    fig.subplots_adjust(left=0.3, right=0.9, top=0.72, bottom=0.14)
    for i, r in rows.iterrows():
        a, b = r[left] * 100, r[right] * 100
        ax.plot([a, b], [i, i], color=AXIS, lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [i], s=64, color=left_colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter([b], [i], s=64, color=right_colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(xmax + 2, i, ratio_fmt.format(r[ratio]), va="center", ha="left", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(rows)), rows.config)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 25, 50, 75, 100] if xmax > 75 else [0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Percent", fontsize=9)
    _style(ax, grid_axis="x")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    handles = [
        Line2D([], [], ls="", marker="o", ms=7, color=left_colour, mec=SURFACE, label=left_name),
        Line2D([], [], ls="", marker="o", ms=7, color=right_colour, mec=SURFACE, label=right_name),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.83), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, title, subtitle)
    fig.text(0.02, 0.012, SOURCE, fontsize=7.2, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, env = load()
    d.to_csv(OUT / "h100_probe_summary.csv", index=False)

    _dumbbell(
        d, "mfu", "counter", S_MFU, S_UTIL, "MFU", "GPU utilization counter (nvidia-smi)",
        "counter_over_mfu", OUT / "fig4_h100_counter_vs_mfu",
        f"On an H100 at 2k–8k tokens, the counter reads {d.counter_over_mfu.min():.1f}–"
        f"{d.counter_over_mfu.max():.1f}× the MFU",
        f"Same training steps. The counter stays at {d.counter.min()*100:.1f}–100% while MFU moves between "
        f"{d.mfu.min()*100:.0f}% and {d.mfu.max()*100:.0f}%.\n"
        "Annex A.2.1 is linear in U, so each ratio is the factor by which reported compute changes.",
        ratio_fmt="×{:.1f}",
    )

    # Recomputation alone: HFU/MFU with checkpointing over the same configuration without it.
    base = d[~d.ckpt].set_index(["seq", "batch"]).hfu_over_mfu
    paired = [r.hfu_over_mfu / base[(r.seq, r.batch)] for r in d[d.ckpt].itertuples()
              if (r.seq, r.batch) in base.index]
    _dumbbell(
        d, "mfu", "hfu", S_MFU, S_HFU, "MFU (Guidelines basis: recomputation excluded)",
        "HFU (FLOP the hardware executes)", "hfu_over_mfu", OUT / "fig5_h100_recomputation",
        f"With checkpointing, the GPU executes {d[d.ckpt].hfu_over_mfu.min()*100-100:.0f}–"
        f"{d[d.ckpt].hfu_over_mfu.max()*100-100:.0f}% more FLOP than the Guidelines count",
        f"Without checkpointing the gap is {d[~d.ckpt].hfu_over_mfu.min()*100-100:.0f}–"
        f"{d[~d.ckpt].hfu_over_mfu.max()*100-100:.0f}%; the recomputation itself adds "
        f"{min(paired)*100-100:.0f}–{max(paired)*100-100:.0f}%.\n"
        "Hardware counts recomputation; Annex A.1 does not. The allowed error margin is 30%.",
        xmax=55, ratio_fmt="×{:.2f}",
    )

    j = d.j_per_model_flop
    summary = {
        "run": RUN.name, "gpu": env["gpu_name"], "driver": env["driver"], "torch": env["torch"],
        "power_limit_w": env["power_limit_w"], "max_sm_clock_mhz": env["max_sm_clock_mhz"],
        "counter_over_mfu_range": [round(d.counter_over_mfu.min(), 2), round(d.counter_over_mfu.max(), 2)],
        "mfu_range_pct": [round(d.mfu.min() * 100, 1), round(d.mfu.max() * 100, 1)],
        "counter_min_pct": round(d.counter.min() * 100, 1),
        "checkpointing_executed_over_model_range": [round(d[d.ckpt].executed_over_model.min(), 3),
                                                    round(d[d.ckpt].executed_over_model.max(), 3)],
        "attention_convention_full_over_half": {int(s): round(g.full_over_half.mean(), 3)
                                               for s, g in d.groupby("seq")},
        "ofu_range_pct": [round(d.ofu.min() * 100, 1), round(d.ofu.max() * 100, 1)],
        "ofu_over_hfu_range": [round((d.ofu / d.hfu).min(), 3), round((d.ofu / d.hfu).max(), 3)],
        "power_w_range": [round(d.power_w.min(), 1), round(d.power_w.max(), 1)],
        "clock_over_max_range": [round(d.clock_over_max.min(), 2), round(d.clock_over_max.max(), 2)],
        "j_per_model_flop_range": [float(f"{j.min():.4g}"), float(f"{j.max():.4g}")],
        "j_per_model_flop_max_over_min": round(j.max() / j.min(), 3),
        "tinyllama_a100_mfu_over_measured_baseline": round(
            0.56 / d[(d.seq == 2048) & (d.batch == 8) & ~d.ckpt].mfu.iloc[0], 2),
    }
    base = d[(d.seq == 2048) & (d.batch == 8) & ~d.ckpt].iloc[0]
    summary["peak_used_tflops"] = PEAK_PCIE_BF16_DENSE / 1e12
    summary["peak_sensitivity"] = {
        name: {
            "baseline_mfu_pct": round(base.mfu * PEAK_PCIE_BF16_DENSE / 1e12 / tf * 100, 1),
            "counter_over_mfu_range": [round((d.counter / (d.mfu * PEAK_PCIE_BF16_DENSE / 1e12 / tf)).min(), 2),
                                       round((d.counter / (d.mfu * PEAK_PCIE_BF16_DENSE / 1e12 / tf)).max(), 2)],
        }
        for name, tf in CANDIDATE_PEAKS_TFLOPS.items()
    }
    (OUT / "h100_probe_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(d[["config", "mfu", "hfu", "counter", "smact", "tensor", "counter_over_mfu", "hfu_over_mfu",
             "full_over_half", "ofu", "power_w"]].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
