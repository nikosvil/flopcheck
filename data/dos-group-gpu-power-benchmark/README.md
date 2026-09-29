# GPU Power Benchmark

A framework for benchmarking GPU power draw, GPU Utilization, and Model FLOPs Utilization (MFU) across different hardware and training configurations.

## Workflow

Run the scripts in order:

```bash
# 1. Collect raw GPU measurements
python benchmark.py

# 2. Aggregate raw CSVs into per-configuration means
python aggregate.py

# 3. Generate analysis tables
python analyze.py

# 4. Per-GPU and per-(dtype, batch) fits with bootstrap R^2/MAPE, plus cross-vendor transfer
python cross_vendor_transfer.py

# 5. Model residuals versus the cross-repeat measurement-noise floor
python noise_floor.py

# 6. Generate figures
python create_figures/regenerate_all.py
```

Steps 3–6 all read from `aggregation_results/` and `benchmark_results/` and are independent of each other.
`analyze.py` prints to stdout; `cross_vendor_transfer.py` writes `cross_vendor_transfer.csv`, `within_cell_all_gpus.csv` and `within_cell_summary.csv`; `noise_floor.py` writes `noise_floor.csv`.

## Configuration

### Benchmark parameters

Edit the `BenchmarkConfig` dataclass in `benchmark.py` to change:
- `models`: list of HuggingFace model IDs to benchmark
- `batch_sizes`: batch sizes to sweep
- `dtype`: precision (`float32`, `float16`, `bfloat16`)
- `context_window`: sequence length
- `gpu_index`: which GPU to benchmark

The Hydra sweep in `AppConfig` controls the cross-product of dtypes, cooldown, warmup, and context window run by default.

### Hardware configurations

`data.py` contains `HW_CONFIGS`, a list of `(aggregation_csv, raw_csv, display_name)` tuples. Add an entry here to include a new GPU in the analysis and figures.

### External power meter

`benchmark.py` can optionally query an external power meter via HTTP. 
Credentials are read from the environment (`POWER_METER_URL`, `POWER_METER_USER`, `POWER_METER_PASSWORD`).
