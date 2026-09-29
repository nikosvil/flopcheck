# H100 probe on Scaleway H100-1-80G, with MobaXterm on Windows

This is the measurement behind Figures 4 and 5 of the preprint *What does "GPU utilization" mean? Testing the 30% accuracy margin for training-compute estimates under the EU AI Act* (https://doi.org/10.5281/zenodo.22881088). **The paper's own run is already in `results/20260916T190612Z/`**, so you don't need an H100 to check the figures: run `python analysis/h100_probe_analysis.py` from the repository root. This guide is only for repeating the measurement yourself.

The steps below use Scaleway and MobaXterm, and the menus, image names and prices are as of September 2026. Any cloud H100 and any SSH client work: MobaXterm is only used to open a terminal and upload two files.

**What it does.** It trains the TinyLlama 1.1B architecture (the model behind Model A in the Commission's June 2025 draft) on one NVIDIA H100, using random tokens at realistic sequence lengths (2k and 8k). On the same training steps it records:
- the `nvidia-smi` utilisation counter
- DCGM tensor-pipe activity, if DCGM can be installed
- HFU: the FLOPs the GPU actually executes, including recomputation
- MFU: the FLOPs the model needs, which is the Guidelines' basis
- power and energy

**Your instance.** Scaleway **H100-1-80G** = 1 × H100 **PCIe**, 80 GB, 24 vCPUs, 240 GiB RAM, in **PAR 2** or **WAW 2**. The script recognises the H100 PCIe and SXM cards automatically. For any other GPU, pass its dense BF16 peak from the vendor datasheet with `--peak-tflops`.

**Time and cost.** Plan on about **one hour** in total. Scaleway bills each resource for **at least 60 minutes**, so expect to pay for one hour. The console shows the hourly price before you confirm.

**How you connect.** You use **MobaXterm** on Windows. Step 2 uses **PowerShell** once, to make the key. Every command after that runs **on the instance**, as `root`, in the MobaXterm session's terminal.

**Where things live on the instance.** The probe folder is **`/root/h100_probe/`** and contains just two files: `h100_probe.py` and `requirements.txt`.

**You do:** the Scaleway account, launching the instance, and deleting it. **The script does:** everything else.

---

## A. Before the session (on your PC, ~15 minutes, ideally a day ahead)

### Step 1. Scaleway account

Sign in at `console.scaleway.com`, add a payment method, and complete any account verification Scaleway asks for. **GPU Instances may stay unavailable until that's done**, which is why it's best to do this a day early.

### Step 2. Make an SSH key (PowerShell, once)

```powershell
ssh-keygen -t ed25519 -f "$HOME\.ssh\h100_probe" -C "h100-probe"
```

Press Enter twice for no passphrase, or set one. Then copy the public half to your clipboard:

```powershell
Get-Content "$HOME\.ssh\h100_probe.pub" | Set-Clipboard
```

This creates `%USERPROFILE%\.ssh\h100_probe` (private, stays on your PC) and `h100_probe.pub` (public, goes to Scaleway).

### Step 3. Add the key to Scaleway (before creating the instance)

In the console: **Organization / Project → SSH keys → Add SSH key**. Paste the key, give it a name, and save.

Keys are injected when an instance is **created**, so this step must come first.

---

## B. The session (~1 hour)

### Step 4. Create the instance

In the console: **CPU & GPU Instances → Create Instance**. Then set:

| Setting | Choose |
|---|---|
| Availability Zone | **PAR 2** (or WAW 2) |
| Instance type | **GPU → H100-1-80G** |
| Image | **Ubuntu Noble GPU OS 13 (NVIDIA)** |
| System volume | Default is fine; Scaleway recommends **125 GB** for GPU images |
| Public IP | **Keep a public IPv4 assigned.** You need it to connect |
| SSH keys | Check that **h100-probe** is ticked |

Click **Create Instance**. When it shows as running, copy its **public IPv4 address**. Below, replace `<IP>` with it. The username is always **`root`**.

### Step 5. Connect with MobaXterm and upload the two files

1. **Session → SSH.** Set **Remote host** to `<IP>`, tick **Specify username** and enter `root`.
2. Open **Advanced SSH settings**, tick **Use private key**, and choose `%USERPROFILE%\.ssh\h100_probe` (the file **without** `.pub`). Click **OK**. The first time, accept the host key.
3. In the session's terminal, create the folder:

```bash
mkdir -p /root/h100_probe
```

4. In MobaXterm's **file browser on the left**, go to `/root/h100_probe/` and drag in **only these two files** from your local copy of the `probe/` folder:
   - `h100_probe.py`
   - `requirements.txt`

Check they arrived:

```bash
ls -l /root/h100_probe
```

### Step 6. Start `tmux` so a dropped connection can't kill the run

In the session's terminal:

```bash
apt-get update && apt-get install -y tmux python3-venv python3-pip
```

```bash
tmux new -s probe
```

### Step 7. Install PyTorch in a virtual environment (~5 minutes)

The Scaleway GPU image has NVIDIA drivers but no PyTorch on the host.

```bash
python3 -m venv /root/venv && source /root/venv/bin/activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

```bash
cd /root/h100_probe && pip install -r requirements.txt
```

Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want a version of **2.3 or newer**, then **True**, then **NVIDIA H100 PCIe** (or similar).

**If the `pip install torch` line fails**, use Plan B at the end of this guide instead.

**If tmux ever reattaches you in a fresh shell,** re-activate the environment first:

```bash
source /root/venv/bin/activate && cd /root/h100_probe
```

### Step 8. DCGM: optional, 10 minutes at most

DCGM adds one extra reading, tensor-pipe activity. The run works fine without it.

```bash
apt-get install -y datacenter-gpu-manager-4-cuda12 && systemctl --now enable nvidia-dcgm && dcgmi discovery -l
```

If it says **"Unable to locate package"** or anything else fails, **skip it**. The script notices by itself.

### Step 9. Preflight (~2 minutes)

```bash
python h100_probe.py preflight
```

It shows the GPU, the peak it'll use (it should say **H100 PCIe**), whether NVML and DCGM work, and runs a 15-second test of the whole pipeline.

**Carry on only if it ends with `READY`.**
- If it says **"GPU not recognised"**, the instance isn't an H100. Look up the card's dense BF16 peak in its datasheet and rerun with `--peak-tflops <value>`.
- If it says **`NOT READY`** for any other reason, the output lists what failed. Fix that before running.

### Step 10. The run (~30 minutes)

```bash
python h100_probe.py run
```

It runs six configurations and prints one line after each. When it's done, it prints the path to a **`.tgz`** file under `/root/h100_probe/results/`.

- **Detach and let it run:** press `Ctrl-b`, then `d`.
- **Come back:** `tmux attach -t probe`.
- **Watch the GPU at the same time:** open a second tmux window with `Ctrl-b` then `c`, and run `watch -n1 nvidia-smi`.
- **Short on time?** `python h100_probe.py run --quick` runs two configurations in about 10 minutes.

---

## C. After the run: don't skip these

### Step 11. Copy the results to your PC

In MobaXterm's **file browser**, go to `/root/h100_probe/results/` (refresh the browser if the folder doesn't show yet). Drag the **`.tgz`** file to your local copy of the `probe/` folder.

**Check the `.tgz` file is on your PC before Step 12.**

### Step 12. Delete the instance, its IP and its volumes

1. In the console: **CPU & GPU Instances**, then pick the zone you used (**PAR 2**) in the drop-down.
2. Click the **⋯** next to the instance, then **Delete**.
3. Type **DELETE** to confirm.
4. **Tick the boxes to delete the associated IP and volumes too.**
5. Click **Delete Instance**.

**Why all four parts matter:** "Power off" stops the instance charge, but its volumes and IP keep billing. **Standby** bills as if the instance were still running. Only deleting the instance *with* its IP and volumes stops everything.

Afterwards, check that **Billing** shows nothing still running.

### Step 13. Analyse the results

Unpack the `.tgz` into `probe/results/`:

```bash
tar -xzf <timestamp>.tgz -C probe/results
```

Open `analysis/h100_probe_analysis.py` and change the run folder on line 54 (`RUN = ROOT / "probe" / "results" / "20260916T190612Z"`) to your own timestamp. Then, from the repository root:

```bash
python analysis/h100_probe_analysis.py
```

Figures 4 and 5 and the summary tables are written to `analysis/out/`.

---

## Plan B: PyTorch from Scaleway's Docker image

Use this only if Step 7 fails. DCGM won't be available this way, so everything else still works but that one reading is missing.

On the instance, inside `tmux`:

```bash
docker run --runtime=nvidia --user root -it --rm -v /root/h100_probe:/probe -w /probe rg.fr-par.scw.cloud/scw-ai/pytorch:latest /bin/bash
```

Now you're inside the container, where the folder appears as `/probe`:

```bash
pip install nvidia-ml-py && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```bash
python h100_probe.py preflight --no-dcgm
```

```bash
python h100_probe.py run --no-dcgm
```

The results land in `/root/h100_probe/results` on the instance itself, so Steps 11–13 are unchanged.

---

## If something goes wrong

| Symptom | What to do |
|---|---|
| **Permission denied (publickey)** | Check **Use private key** points to `%USERPROFILE%\.ssh\h100_probe`, the file without `.pub`. If it does, the key wasn't added before the instance was created (Step 3): delete the instance and create a new one. |
| Connection hangs | Check the instance is running and has a **public IPv4**, and that you're using that address. |
| No H100-1-80G offered | Try **WAW 2**, or finish account verification (Step 1). |
| `python: can't open file 'h100_probe.py'` | You're in the wrong folder. Run `cd /root/h100_probe`, and check with `ls` that both files are there. |
| Preflight: **GPU not recognised** | Not an H100. Rerun with `--peak-tflops <dense BF16 peak>` from the card's datasheet. |
| Preflight: DCGM unavailable | Carry on. |
| `torch.compile` errors | The script falls back to eager mode by itself. You can also add `--no-compile`. |
| Out of memory | The script halves the batch once and retries. |
| Connection drops mid-run | Reconnect the MobaXterm session, then `tmux attach -t probe`. The run carries on. |
| A configuration takes much longer than 5 minutes | Check `nvidia-smi` in a second tmux window. If utilisation is 0%, press `Ctrl-c` and check the script's output for the error. |
| You forget Step 12 | It keeps billing every hour. Delete it (with IP and volumes) as soon as you notice. |

## What's in the results

Everything is in `/root/h100_probe/results/<timestamp>/`, packed into a `.tgz` next to it:

- `env.json`: the GPU, driver and PyTorch version
- `results.jsonl`: one line per configuration
- `nvml_*.csv`: the raw NVML samples
- `dcgm_*.log`: the raw DCGM log, if DCGM was available

Nothing leaves the instance except the `.tgz` you copy back.
