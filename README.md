# What does "GPU utilization" mean? Code and data

Code and data for the preprint:

> Vilanakis, N. D. (2026). *What does "GPU utilization" mean? Testing the 30% accuracy margin for training-compute estimates under the EU AI Act.* Preprint. Zenodo. 10.5281/zenodo.22881088

Under Article 51(2) of the EU AI Act, a general-purpose AI model trained with more than 10²⁵ FLOP is presumed to carry systemic risk. The Commission's Guidelines (18 July 2025, Annex A.2) let providers estimate training compute with a hardware-based method, `C = N × L × H × U`, or an architecture-based method, `C ≈ 6 × P × D`, and accept any estimate that is accurate within 30%. This repository measures how much the undefined terms, "GPU utilisation" `U` and peak performance `H`, change the result.

## Contents

| Folder | What it is |
|---|---|
| `flopcheck/` | Reference implementation of both Annex A.2 methods. It reproduces the Commission's worked examples and has tests. |
| `probe/` | The H100 measurement script (`h100_probe.py`) and the raw results from one NVIDIA H100 PCIe, 16 September 2026. |
| `analysis/` | Scripts that turn the measurements into the paper's figures and tables, with their outputs in `analysis/out/`. |
| `data/dos-group-gpu-power-benchmark/` | Third-party benchmark data (Enskat, Wiesner & Kao, MIT licence), re-analysed in the paper. See its own `README.md` and `SOURCE_COMMIT.txt`. |

## Reproducing the results

Python 3.10 or newer.

**flopcheck** (no dependencies apart from pytest):

```bash
cd flopcheck
pip install -e ".[dev]"
pytest
python -m flopcheck.report
```

**Figures 1–5** (needs pandas and matplotlib; run from the repository root):

```bash
pip install pandas matplotlib
python analysis/reanalysis_dos_group.py
python analysis/h100_probe_analysis.py
```

| Script | Input | Output in `analysis/out/` |
|---|---|---|
| `reanalysis_dos_group.py` | `data/dos-group-gpu-power-benchmark/` | `fig1_util_vs_mfu`, `fig2_gap_at_batch128`, `fig3_flop_count_methods`, `fp32_peak_report.csv`, `summary.json` |
| `h100_probe_analysis.py` | `probe/results/20260916T190612Z/` | `fig4_h100_counter_vs_mfu`, `fig5_h100_recomputation`, `h100_probe_summary.csv`, `h100_probe_summary.json` |

`analysis/REANALYSIS.md` and `analysis/H100_PROBE.md` describe the methods and the results in detail.

**Re-running the H100 measurement** needs an NVIDIA H100 and PyTorch 2.3 or newer. `probe/README.md` walks through it step by step on a rented cloud instance (about one hour).

```bash
cd probe
pip install -r requirements.txt
python h100_probe.py preflight
python h100_probe.py run
```

## The H100 results

| Folder | What it holds |
|---|---|
| `probe/results/20260916T190612Z/` | The run used in the paper: six configurations (sequence length 2,048 or 8,192; with and without activation checkpointing) |
| `probe/results/preflight/` | The 15-second preflight check |

Each run folder contains `env.json` (GPU, driver and software versions), `results.jsonl` (one line per configuration), `nvml_*.csv` (raw NVML samples) and `dcgm_*.log` (raw DCGM output).

## Citation

If you use this code or data, please cite the preprint above. GitHub's "Cite this repository" button uses `CITATION.cff`.

## Licence

MIT, see `LICENSE`. The data in `data/dos-group-gpu-power-benchmark/` keeps its own MIT licence, © Niklas Enskat, Philipp Wiesner and Odej Kao.
