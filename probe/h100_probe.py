#!/usr/bin/env python3
"""h100_probe -- one GPU, one training run, every reading of "GPU utilisation".

Trains a TinyLlama-architecture decoder (1.1B parameters, the model behind
Model A of the Commission's June 2025 draft Guidelines) on random tokens at
realistic sequence lengths, and records on the *same* training steps:

  * throughput, and MFU on the model-FLOP basis of Annex A.1 (activation
    recomputation excluded);
  * HFU -- FLOPs actually executed, counted by torch.utils.flop_counter, which
    includes recomputation when activation checkpointing is on;
  * the NVML utilisation counter (the number nvidia-smi shows);
  * DCGM tensor-pipe activity, SM activity and SM clock, from which OFU
    (Pedersen et al. 2026, arXiv:2605.20799) is computed;
  * power and energy (NVML's cumulative energy counter).

Random tokens are deliberate: the FLOPs of a training step do not depend on
the token values, and it removes the data pipeline entirely.

Usage
  python h100_probe.py preflight     ~2 min   checks GPU, NVML, DCGM; tiny benchmark
  python h100_probe.py run           ~35 min  the full sweep (6 configurations)
  python h100_probe.py run --quick   ~10 min  two configurations
  python h100_probe.py dry-run       CPU      tiny model; checks the code path only

Everything is written to results/<UTC timestamp>/ and packed into a .tgz.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Pure helpers (no torch / NVML needed -- unit-tested locally)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arch:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ffn: int
    vocab: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


#: TinyLlama 1.1B (arXiv:2401.02385): 22 layers, d 2048, 32 heads, 4 KV heads,
#: SwiGLU 5632, vocabulary 32 000, untied output head.
TINYLLAMA = Arch(22, 2048, 32, 4, 5632, 32000)
#: For the CPU dry run only.
TOY = Arch(2, 128, 4, 2, 256, 512)


def param_counts(a: Arch) -> dict[str, int]:
    hd = a.head_dim
    attn = a.d_model * a.n_heads * hd * 2 + a.d_model * a.n_kv_heads * hd * 2
    mlp = 3 * a.d_model * a.d_ffn
    embed = a.vocab * a.d_model
    head = a.vocab * a.d_model
    total = embed + a.n_layers * (attn + mlp + 2 * a.d_model) + a.d_model + head
    return {"total": total, "embedding": embed, "matmul": a.n_layers * (attn + mlp) + head}


def model_flops_per_token(a: Arch, seq: int, causal_half: bool = False) -> float:
    """Model FLOPs per training token, recomputation excluded (Annex A.1 basis).

    6N over the matmul weights (the input embedding is a lookup, not a matmul)
    plus the attention term 12*L*H*hd*T, the PaLM convention. Some codebases
    halve the attention term for causal masking; ``causal_half`` gives that
    variant, because even "MFU" has more than one convention.
    """
    attn = 12.0 * a.n_layers * a.n_heads * a.head_dim * seq
    return 6.0 * param_counts(a)["matmul"] + (attn / 2 if causal_half else attn)


#: (substring of the NVML device name, label, dense BF16 peak FLOP/s,
#:  tensor-core max clock in MHz for OFU or None if not established).
#: H100 SXM: 989.4 TFLOP/s dense BF16; tensor cores boost to 1 830 MHz
#: (Pedersen et al. 2026). H100 PCIe: 114 SMs x 4096 x 1620 MHz = 756.45 TFLOP/s
#: dense BF16, tensor cores boost to 1 620 MHz (NVIDIA developer forum).
GPU_PEAKS = [
    ("H100 80GB HBM3", "H100 SXM", 989.4e12, 1830),
    ("H100 SXM", "H100 SXM", 989.4e12, 1830),
    ("H100 PCIe", "H100 PCIe", 756.45e12, 1620),
]


def detect_peak(name: str, tflops: float | None = None, tc_mhz: float | None = None):
    """Return (label, peak FLOP/s, tensor-core max MHz) for a device name."""
    if tflops is not None:
        return ("override", tflops * 1e12, tc_mhz)
    for key, label, peak, mhz in GPU_PEAKS:
        if key.lower() in name.lower():
            return (label, peak, tc_mhz if tc_mhz is not None else mhz)
    return (None, None, None)


#: DCGM fields requested, in order: (field id, key). 1002/1004 are ratios 0..1,
#: 100 is MHz, 155 is W, 203 is a percentage.
DCGM_FIELDS = [
    (1002, "sm_active"),
    (1004, "tensor_active"),
    (100, "sm_clock_mhz"),
    (155, "power_w"),
    (203, "gpu_util_pct"),
]


def parse_dcgm_dmon(text: str, n_fields: int) -> list[list[float]]:
    """Parse `dcgmi dmon` output into rows of floats, in the requested order.

    Data lines look like ``GPU 0   0.912   0.521   1980   612.4   100``.
    Headers, unit lines and rows containing N/A are skipped.
    """
    rows = []
    for line in text.splitlines():
        t = line.split()
        if len(t) >= 2 + n_fields and t[0] == "GPU":
            try:
                rows.append([float(v) for v in t[2 : 2 + n_fields]])
            except ValueError:
                continue
    return rows


def ofu_from_samples(tensor_active: list[float], sm_clock: list[float], tc_max_mhz: float) -> float:
    """OFU = mean(tensor-pipe activity x SM clock / tensor-core max clock)."""
    pairs = list(zip(tensor_active, sm_clock))
    return sum(t * c / tc_max_mhz for t, c in pairs) / len(pairs) if pairs else float("nan")


def summarise(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1)))]  # noqa: E731
    return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s), "p10": q(0.1), "p90": q(0.9)}


@dataclass(frozen=True)
class Config:
    seq: int
    batch: int
    ckpt: bool


SWEEP = [
    Config(2048, 8, False),   # baseline
    Config(2048, 8, True),    # same, full activation checkpointing -> HFU vs MFU
    Config(2048, 2, False),   # small batch -> lower MFU, counter stays high
    Config(8192, 2, False),   # long context -> attention term matters
    Config(8192, 2, True),
    Config(2048, 16, True),   # larger batch made possible by checkpointing
]
QUICK = SWEEP[:2]


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class NVMLSampler(threading.Thread):
    """Polls utilisation, power and SM clock at ~10 Hz."""

    def __init__(self, nvml, handle, path: Path, period: float = 0.1):
        super().__init__(daemon=True)
        self.nvml, self.h, self.path, self.period = nvml, handle, path, period
        self.rows: list[tuple[float, float, float, float]] = []
        self._halt = threading.Event()

    def run(self):
        n, h = self.nvml, self.h
        t0 = time.perf_counter()
        while not self._halt.is_set():
            try:
                util = n.nvmlDeviceGetUtilizationRates(h).gpu
                power = n.nvmlDeviceGetPowerUsage(h) / 1000.0
                clk = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
                self.rows.append((time.perf_counter() - t0, float(util), power, float(clk)))
            except Exception:  # noqa: BLE001 -- keep sampling
                pass
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "util_pct", "power_w", "sm_clock_mhz"])
            w.writerows(self.rows)
        return self.rows


class DCGMSampler:
    """Runs `dcgmi dmon` in the background and parses it afterwards."""

    def __init__(self, path: Path, gpu: int = 0, period_ms: int = 100):
        self.path = path
        self.cmd = ["dcgmi", "dmon", "-i", str(gpu), "-e",
                    ",".join(str(f) for f, _ in DCGM_FIELDS), "-d", str(period_ms)]

    def start(self):
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(self.cmd, stdout=self.f, stderr=subprocess.STDOUT, text=True)

    def stop(self) -> dict[str, list[float]]:
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.f.close()
        rows = parse_dcgm_dmon(self.path.read_text(errors="ignore"), len(DCGM_FIELDS))
        return {k: [r[i] for r in rows] for i, (_, k) in enumerate(DCGM_FIELDS)}


def dcgm_check() -> tuple[bool, str]:
    if shutil.which("dcgmi") is None:
        return False, "dcgmi not installed"
    try:
        out = subprocess.run(["dcgmi", "dmon", "-i", "0", "-e", "1004,100", "-c", "5", "-d", "200"],
                             capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "dcgmi timed out (is nv-hostengine running?)"
    text = out.stdout + out.stderr
    rows = parse_dcgm_dmon(text, 2)
    if rows:
        return True, f"{len(rows)} samples"
    return False, text.strip().splitlines()[-1] if text.strip() else "no output"


# ---------------------------------------------------------------------------
# Model and training step (torch imported lazily)
# ---------------------------------------------------------------------------


def build_model(a: Arch, ckpt: bool):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    try:
        q = torch.zeros(1, 4, 2, 8)
        F.scaled_dot_product_attention(q, q[:, :2], q[:, :2], is_causal=True, enable_gqa=True)
        gqa_native = True
    except (TypeError, RuntimeError):
        gqa_native = False
    # torch.utils.flop_counter asserts equal query and key/value head counts, so
    # while counting FLOPs the key/value heads are expanded explicitly. The
    # attention arithmetic is the same; only the timed steps use native GQA.
    opts = {"repeat_kv": not gqa_native}

    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.w = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            hd = a.head_dim
            self.n1, self.n2 = RMSNorm(a.d_model), RMSNorm(a.d_model)
            self.wq = nn.Linear(a.d_model, a.n_heads * hd, bias=False)
            self.wk = nn.Linear(a.d_model, a.n_kv_heads * hd, bias=False)
            self.wv = nn.Linear(a.d_model, a.n_kv_heads * hd, bias=False)
            self.wo = nn.Linear(a.n_heads * hd, a.d_model, bias=False)
            self.w1 = nn.Linear(a.d_model, a.d_ffn, bias=False)
            self.w3 = nn.Linear(a.d_model, a.d_ffn, bias=False)
            self.w2 = nn.Linear(a.d_ffn, a.d_model, bias=False)

        def attn(self, x):
            B, T, _ = x.shape
            hd = a.head_dim
            q = self.wq(x).view(B, T, a.n_heads, hd).transpose(1, 2)
            k = self.wk(x).view(B, T, a.n_kv_heads, hd).transpose(1, 2)
            v = self.wv(x).view(B, T, a.n_kv_heads, hd).transpose(1, 2)
            if not opts["repeat_kv"]:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                r = a.n_heads // a.n_kv_heads
                o = F.scaled_dot_product_attention(q, k.repeat_interleave(r, 1), v.repeat_interleave(r, 1), is_causal=True)
            return self.wo(o.transpose(1, 2).reshape(B, T, -1))

        def forward(self, x):
            x = x + self.attn(self.n1(x))
            h = self.n2(x)
            return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(a.vocab, a.d_model)
            self.blocks = nn.ModuleList(Block() for _ in range(a.n_layers))
            self.norm = RMSNorm(a.d_model)
            self.head = nn.Linear(a.d_model, a.vocab, bias=False)
            self.opts = opts

        def forward(self, x, y):
            h = self.emb(x)
            for b in self.blocks:
                h = checkpoint(b, h, use_reentrant=False) if ckpt else b(h)
            logits = self.head(self.norm(h))
            return F.cross_entropy(logits.float().view(-1, a.vocab), y.reshape(-1))

    return Model(), gqa_native


def count_flops_per_step(model, x, y, device_type, amp_dtype) -> float:
    """FLOPs of one eager forward+backward, as executed (HFU basis)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    previous = model.opts["repeat_kv"]
    model.opts["repeat_kv"] = True
    try:
        with FlopCounterMode(display=False) as fc:
            with torch.autocast(device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(x, y)
            loss.backward()
    finally:
        model.opts["repeat_kv"] = previous
        model.zero_grad(set_to_none=True)
    return float(fc.get_total_flops())


def run_config(cfg: Config, a: Arch, env: dict, outdir: Path, *, device: str, compile_model: bool,
               warmup_s: float, measure_s: float, nvml=None, handle=None, use_dcgm=False) -> dict:
    import torch

    torch.manual_seed(0)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_dtype = torch.bfloat16 if device_type == "cuda" else None
    model, gqa_native = build_model(a, cfg.ckpt)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device_type == "cuda"))
    tok = torch.randint(0, a.vocab, (cfg.batch, cfg.seq + 1), device=device)
    x, y = tok[:, :-1].contiguous(), tok[:, 1:].contiguous()
    tokens_per_step = cfg.batch * cfg.seq

    # Executed FLOPs, counted once on the eager model (fall back to batch 1 on OOM).
    try:
        counted = count_flops_per_step(model, x, y, device_type, amp_dtype)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        counted = cfg.batch * count_flops_per_step(model, x[:1], y[:1], device_type, amp_dtype)

    fwd = model
    compiled = False
    if compile_model:
        try:
            fwd = torch.compile(model)
            compiled = True
        except Exception as e:  # noqa: BLE001
            print(f"  torch.compile unavailable ({e!r}); running eager")

    def step():
        with torch.autocast(device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = fwd(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    sync = torch.cuda.synchronize if device_type == "cuda" else (lambda: None)

    t0, n = time.perf_counter(), 0
    try:
        step()  # the first call is where torch.compile actually compiles
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:  # noqa: BLE001
        if not compiled:
            raise
        print(f"  torch.compile failed on first step ({type(e).__name__}); falling back to eager")
        fwd, compiled = model, False
        opt.zero_grad(set_to_none=True)
        step()
    n = 1
    while time.perf_counter() - t0 < warmup_s or n < 3:
        step()
        n += 1
    sync()

    tag = f"s{cfg.seq}_b{cfg.batch}_{'ckpt' if cfg.ckpt else 'nockpt'}"
    nv = NVMLSampler(nvml, handle, outdir / f"nvml_{tag}.csv") if nvml else None
    dc = DCGMSampler(outdir / f"dcgm_{tag}.log") if use_dcgm else None
    e0 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle) if nvml else None
    if nv:
        nv.start()
    if dc:
        dc.start()

    t0, steps = time.perf_counter(), 0
    while time.perf_counter() - t0 < measure_s or steps < 3:
        step()
        steps += 1
    sync()
    elapsed = time.perf_counter() - t0

    nv_rows = nv.stop() if nv else []
    dc_cols = dc.stop() if dc else {}
    e1 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle) if nvml else None

    tps = steps * tokens_per_step / elapsed
    mf = model_flops_per_token(a, cfg.seq)
    peak = env.get("peak_flops") or 1.0
    rec = {
        "config": asdict(cfg), "compiled": compiled, "gqa_native": gqa_native,
        "steps": steps, "elapsed_s": elapsed, "tokens_per_s": tps,
        "model_flops_per_token": mf,
        "model_flops_per_token_causal_half": model_flops_per_token(a, cfg.seq, causal_half=True),
        "executed_flops_per_token": counted / tokens_per_step,
        "peak_flops": env.get("peak_flops"),
        "mfu": tps * mf / peak,
        "mfu_causal_half": tps * model_flops_per_token(a, cfg.seq, causal_half=True) / peak,
        "hfu": tps * (counted / tokens_per_step) / peak,
    }
    if nv_rows:
        rec["nvml_util_pct"] = summarise([r[1] for r in nv_rows])
        rec["nvml_power_w"] = summarise([r[2] for r in nv_rows])
        rec["nvml_sm_clock_mhz"] = summarise([r[3] for r in nv_rows])
    if e0 is not None and e1 is not None:
        joules = (e1 - e0) / 1000.0
        rec["energy_j"] = joules
        rec["joules_per_model_flop"] = joules / (steps * tokens_per_step * mf)
    if dc_cols and dc_cols.get("tensor_active"):
        for k, v in dc_cols.items():
            rec[f"dcgm_{k}"] = summarise(v)
        if env.get("tc_max_mhz"):
            rec["ofu"] = ofu_from_samples(dc_cols["tensor_active"], dc_cols["sm_clock_mhz"], env["tc_max_mhz"])

    del model, opt, fwd
    if device_type == "cuda":
        torch.cuda.empty_cache()
    return rec


def fmt_row(r: dict) -> str:
    c = r["config"]
    util = r.get("nvml_util_pct", {}).get("mean", float("nan"))
    ofu = r.get("ofu", float("nan"))
    return (f"  seq {c['seq']:>5}  batch {c['batch']:>3}  ckpt {str(c['ckpt']):<5}  "
            f"{r['tokens_per_s']:>9.0f} tok/s   MFU {r['mfu']*100:5.1f}%   HFU {r['hfu']*100:5.1f}%   "
            f"counter {util:5.1f}%   OFU {ofu*100:5.1f}%   counter/MFU {util/100/r['mfu']:4.2f}x")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def gpu_env(args) -> tuple[dict, object, object]:
    import torch

    env = {"python": platform.python_version(), "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "utc": datetime.now(timezone.utc).isoformat()}
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        h = nvml.nvmlDeviceGetHandleByIndex(0)
        name = nvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        drv = nvml.nvmlSystemGetDriverVersion()
        env.update({
            "gpu_name": name,
            "driver": drv.decode() if isinstance(drv, bytes) else drv,
            "max_sm_clock_mhz": nvml.nvmlDeviceGetMaxClockInfo(h, nvml.NVML_CLOCK_SM),
            "power_limit_w": nvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            "gpu_count": nvml.nvmlDeviceGetCount(),
        })
    except Exception as e:  # noqa: BLE001
        nvml, h = None, None
        env["nvml_error"] = repr(e)
        env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    label, peak, mhz = detect_peak(env["gpu_name"], args.peak_tflops, args.tc_max_mhz)
    env.update({"peak_label": label, "peak_flops": peak, "tc_max_mhz": mhz})
    ok, msg = (dcgm_check() if not args.no_dcgm else (False, "disabled by --no-dcgm"))
    env.update({"dcgm_ok": ok, "dcgm_msg": msg})
    try:
        env["nvidia_smi_q"] = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK,POWER"],
                                             capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        pass
    return env, nvml, h


def new_outdir() -> Path:
    d = HERE / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d.mkdir(parents=True, exist_ok=True)
    return d


def pack(outdir: Path) -> Path:
    tgz = outdir.with_suffix(".tgz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(outdir, arcname=outdir.name)
    return tgz


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_preflight(args) -> int:
    import torch

    print("== preflight ==")
    problems = []
    if not torch.cuda.is_available():
        print("  CUDA not available to PyTorch.  -> STOP. See README, step 6.")
        return 1
    env, nvml, h = gpu_env(args)
    v = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    print(f"  GPU            {env['gpu_name']}  (driver {env.get('driver', '?')}, {env.get('gpu_count', '?')} GPU(s))")
    print(f"  PyTorch        {env['torch']}  CUDA {env['torch_cuda']}")
    print(f"  Peak used      {env['peak_label']}: "
          f"{(env['peak_flops'] or 0)/1e12:.1f} TFLOP/s dense BF16, tensor-core clock {env['tc_max_mhz']} MHz")
    print(f"  NVML           {'ok' if nvml else 'MISSING: ' + env.get('nvml_error', '')}")
    print(f"  DCGM profiling {'ok (' + env['dcgm_msg'] + ')' if env['dcgm_ok'] else 'unavailable: ' + env['dcgm_msg']}")
    if env["peak_flops"] is None:
        problems.append("GPU not recognised as H100 SXM/PCIe: MFU needs --peak-tflops from the datasheet")
    if v < (2, 3):
        problems.append("PyTorch older than 2.3: upgrade (README step 6)")
    if not nvml:
        problems.append("NVML bindings missing: pip install nvidia-ml-py")
    if not torch.cuda.is_bf16_supported():
        problems.append("BF16 not supported")
    if problems:
        print("\n  NOT READY:")
        for p in problems:
            print("   -", p)
        return 1

    print("\n  Micro-benchmark (seq 2048, batch 2, eager, 15 s) ...")
    outdir = HERE / "results" / "preflight"
    outdir.mkdir(parents=True, exist_ok=True)
    r = run_config(Config(2048, 2, False), TINYLLAMA, env, outdir, device="cuda:0", compile_model=False,
                   warmup_s=5, measure_s=15, nvml=nvml, handle=h, use_dcgm=env["dcgm_ok"])
    print(fmt_row(r))
    (outdir / "preflight.json").write_text(json.dumps({"env": env, "result": r}, indent=2, default=str))
    print("\n  READY." + ("" if env["dcgm_ok"] else "  (Without DCGM: OFU will be missing; everything else works.)"))
    print("  Next:  python h100_probe.py run")
    return 0


def cmd_run(args) -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available. Run preflight first.")
        return 1
    env, nvml, h = gpu_env(args)
    if env["peak_flops"] is None:
        print(f"Unrecognised GPU '{env['gpu_name']}'. Pass --peak-tflops (dense BF16) from its datasheet.")
        return 1
    sweep = QUICK if args.quick else SWEEP
    per = args.warmup + args.measure + (90 if not args.no_compile else 0)
    print(f"== run: {len(sweep)} configurations, about {len(sweep) * per / 60:.0f} minutes ==")
    print(f"   {env['gpu_name']} | peak {env['peak_flops']/1e12:.1f} TFLOP/s | DCGM {'on' if env['dcgm_ok'] else 'off'}")
    outdir = new_outdir()
    (outdir / "env.json").write_text(json.dumps(env, indent=2, default=str))
    with open(outdir / "results.jsonl", "w") as out:
        for cfg in sweep:
            for attempt in range(2):
                try:
                    r = run_config(cfg, TINYLLAMA, env, outdir, device="cuda:0",
                                   compile_model=not args.no_compile, warmup_s=args.warmup,
                                   measure_s=args.measure, nvml=nvml, handle=h, use_dcgm=env["dcgm_ok"])
                    out.write(json.dumps(r, default=str) + "\n")
                    out.flush()
                    print(fmt_row(r))
                    break
                except torch.cuda.OutOfMemoryError:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if attempt == 0 and cfg.batch > 1:
                        print(f"  OOM at batch {cfg.batch}; retrying at batch {cfg.batch // 2}")
                        cfg = Config(cfg.seq, cfg.batch // 2, cfg.ckpt)
                    else:
                        print(f"  OOM; skipping {cfg}")
                        break
                except Exception as e:  # noqa: BLE001
                    print(f"  FAILED {cfg}: {e!r}")
                    out.write(json.dumps({"config": asdict(cfg), "error": repr(e)}) + "\n")
                    break
    tgz = pack(outdir)
    print(f"\nDone. Results packed into:\n  {tgz}")
    print("Copy it back (README step 10), then TERMINATE the instance (step 11).")
    return 0


def cmd_dry_run(args) -> int:
    import torch  # noqa: F401

    print("== dry run on CPU with a toy model (checks the code path, not the GPU) ==")
    env = {"peak_flops": None, "tc_max_mhz": None}
    outdir = HERE / "results" / "dry-run"
    outdir.mkdir(parents=True, exist_ok=True)
    for ckpt in (False, True):
        r = run_config(Config(64, 2, ckpt), TOY, env, outdir, device="cpu", compile_model=False,
                       warmup_s=0, measure_s=0.5)
        ratio = r["executed_flops_per_token"] / r["model_flops_per_token"]
        print(f"  ckpt={ckpt!s:<5}  steps {r['steps']:>3}  executed/model FLOPs {ratio:.3f}")
    print("  OK")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["preflight", "run", "dry-run"])
    p.add_argument("--quick", action="store_true", help="two configurations instead of six")
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile (slower, lower MFU)")
    p.add_argument("--no-dcgm", action="store_true", help="skip DCGM even if available")
    p.add_argument("--warmup", type=float, default=45.0, help="warm-up seconds per configuration")
    p.add_argument("--measure", type=float, default=120.0, help="measured seconds per configuration")
    p.add_argument("--peak-tflops", type=float, default=None, help="override dense BF16 peak (datasheet)")
    p.add_argument("--tc-max-mhz", type=float, default=None, help="override tensor-core max clock for OFU")
    args = p.parse_args(argv)
    return {"preflight": cmd_preflight, "run": cmd_run, "dry-run": cmd_dry_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
