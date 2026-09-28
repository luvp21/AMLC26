# PARAM Shavak — Nirma University GPU server

Real specs below, confirmed via SSH access (this replaces earlier guessed specs — the
box turned out to have a different CPU/RAM/OS than assumed before access was granted).

| | |
|---|---|
| Hostname | `mtechcse-Precision-7920-Tower` (Dell Precision 7920 Tower) |
| CPU | Intel Xeon Platinum 8180, 28 physical cores / 56 threads |
| GPU | 1× NVIDIA Quadro GP100, 16GB VRAM, Pascal, compute capability 6.0 |
| RAM | ~250 GiB |
| OS | Ubuntu, Python 3.12.3 system Python |
| Storage | 3.6TB root filesystem, **only ~104GB free (97% used)** — see budget note below |
| NVIDIA driver | 580.178.04 (working; reports CUDA Version 13.0 via `nvidia-smi`) |
| CUDA Toolkit | **Not installed** — no `nvcc`. Driver ≠ toolkit; don't conflate the two. |

## What this box is good for

- **CPU-heavy work, unambiguously** — 56 threads / 250GB RAM crushes both a laptop and an
  AWS `g5.xlarge` (4 vCPU/16GB) for feature engineering, TF-IDF fitting, GBM training
  (`src/models/baseline_gbm.py`), and OCR batch jobs. Run these here, free, during the whole
  prep window.
- **A real second GPU, with the same Pascal caveats as before** — see below.

## Connecting

1. **VPN** (Palo Alto GlobalProtect via OpenConnect — the official Nirma portal has no native
   Linux GlobalProtect client):
   ```bash
   sudo openconnect --protocol=gp -u <your-nirma-id> vpn1.nirmauni.ac.in
   ```
   Keep this process running (foreground or backgrounded) for the whole session — it brings
   up a `tun0` interface with a VPN-assigned address. Fallback gateway: `vpn2.nirmauni.ac.in`.
2. **SSH**, once the VPN is up:
   ```bash
   ssh <your-username>@10.1.19.16
   ```
   Only connect to this known, authorized host — don't probe other addresses on the
   university network.

## Pascal caveats — live-verified 24 Sept, hours before the sprint

The GP100 (~9.3 TFLOPS FP32, comparable to or ahead of a T4 on raw compute) predates
things modern fine-tuning recipes assume. Here's what's actually confirmed, not guessed:

1. **The default pip-installed torch is unusable — pin the version.** `pip install torch`
   (even with `--index-url .../cu118`) currently resolves to a build with **no sm_60 (Pascal)
   kernels at all** — not a bf16/tensor-core gap, total CUDA failure (`no kernel image is
   available for execution on the device` on the first matmul). Recent PyTorch releases have
   dropped Pascal from their default wheels entirely, regardless of which CUDA-tagged index
   you point at. **Confirmed working: `torch==2.5.1` from
   `https://download.pytorch.org/whl/cu121`** — fp16 matmul verified working
   (20x 4096² in 0.36s). `setup_env.sh` now pins this. If a future dependency bump breaks it
   again, the fix is to pin an *older torch version*, not a different CUDA index — check by
   running `test_gpu.py`'s fp16 matmul section after any torch upgrade.
2. **No tensor cores.** fp16 works and roughly doubles throughput over fp32, but you won't
   get the 4–8x speedups tensor-core GPUs (T4/A10G/A100) see on fp16/int8 matmuls. Expect
   fine-tuning to be noticeably slower here than on the AWS `g5.xlarge` per wall-clock hour.
3. **bf16: `torch.cuda.is_bf16_supported()` reports `True`**, which was unexpected for
   Pascal — but this is an unverified capability flag, not a proven fast path (unlike fp16,
   which was directly matmul-tested). **Default to `fp16` anyway** (`finetune_lora.py --dtype
   fp16`) since that's the one actually confirmed to execute correctly.
4. **bitsandbytes 4-bit (QLoRA): still unknown, not because of the GPU.** The test got
   blocked by a network issue (see below) before it could actually exercise 4-bit
   quantization. **Decision: default to `finetune_lora.py --no_4bit` (plain LoRA, fp16, no
   quantization) for the sprint** — 16GB VRAM still fits DeBERTa-base/RoBERTa-large or a
   2B-class VLM unquantized. Only worth re-testing 4-bit if there's spare time; don't block
   the sprint start on it.
5. **Large binary downloads (HF model weights) get reset — this is a firewall, not
   flakiness.** Every attempt to pull `distilbert-base-uncased`'s ~268MB weights directly on
   this box — via `huggingface_hub`, with `xet` disabled, and via raw `curl` with 8 retries —
   failed identically: connection reset after ~1KB, every single time. Config/JSON files
   (small) download fine. This looks like the university firewall blocking large binary
   transfers to HF's CDN specifically, not congestion. **Rule for the whole sprint: never
   download model weights directly on PARAM Shavak. Download on your laptop (normal
   internet) and `rsync` them in** — see below.
6. **It's a shared university machine.** Check who else is logged in / running jobs
   (`nvidia-smi`, `w`) before assuming the full 16GB VRAM and all 56 threads are yours during
   the sprint. Never kill a process you didn't start. (Confirmed 24 Sept: one other idle
   desktop session, ~450MiB GPU memory, 0% util — harmless.)

## Getting code AND model weights onto the box

No git remote is set up for this repo yet, so the fastest path right now is
`rsync` from your laptop (same pattern as `infra/aws/README.md`'s EC2 sync),
once the VPN is up. **Run this from a terminal on your laptop — not from
inside the SSH session on PARAM Shavak** (running it there tries to rsync a
path that only exists on your laptop and just fails on the SSH handshake):

```bash
rsync -avz --exclude data --exclude runs --exclude .git \
    /home/luv/Projects/mlchallange/ <your-username>@10.1.19.16:~/projects/mlchallange/
```

Re-run the same command to push code updates during the sprint — it only
transfers changed files. If you'd rather use git (better for tracking which
version is running where across two boxes), push this repo to a remote
(GitHub private repo, or any git host you have) and `git clone`/`git pull`
from PARAM Shavak instead; ask if you want help setting that up.

**Model weights work the same way and it's not optional** — see the network
caveat above. Download on your laptop first, then sync the HF cache over:

```bash
# on your laptop:
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('MODEL_NAME')"
rsync -avz ~/.cache/huggingface/ <your-username>@10.1.19.16:~/.cache/huggingface/
```

Then on PARAM Shavak, force offline mode so nothing tries the campus link again:
```bash
HF_HUB_OFFLINE=1 python -m src.models.finetune_lora ...
```

**Worth doing once, since you'll rsync repeatedly all sprint** — set up an SSH
key so you're not retyping your password every time:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/param_shavak -N ""
ssh-copy-id -i ~/.ssh/param_shavak.pub <your-username>@10.1.19.16
```
Then add `-e "ssh -i ~/.ssh/param_shavak"` to the rsync commands above (or an
entry in `~/.ssh/config`) for passwordless syncs.

## Storage budget — this is tight, plan around it

**~104GB is free** on a 3.6TB disk that's already 97% full — confirmed via `df -h ~` on
24 Sept. You've explicitly cleared using up to ~100GB of that for this project, so no
need to be overly conservative within that budget — just don't blow past it:

- Before downloading anything large (HF model weights, checkpoints, datasets), run `df -h ~`
  and estimate the size first. If a download could hit tens of GB, stop and confirm before
  running it.
- Use `~/models`, `~/datasets`, `~/projects` for your own files; don't duplicate the same
  model/dataset across multiple locations.
- Start with a smaller model to validate the pipeline before pulling the full-size one.
- Never clean up, delete, or reorganize files/directories you didn't create — this is a
  shared machine with other users' data on it.

## Environment policy — user-level only, no sudo/system installs

The university explicitly disallows installing software without admin permission, and this
account has no sudo. Concretely:

- No `sudo pip install`, `sudo apt install/remove`, driver/CUDA/kernel-module changes, or
  Docker configuration changes — ever, unless a university admin explicitly authorizes it.
- All Python work happens in a **user-level venv**, not system Python and not conda (keeps
  disk usage minimal given the storage budget above):
  ```bash
  python3 -m venv ~/venvs/ml
  source ~/venvs/ml/bin/activate
  ```
- Directory layout:
  ```
  ~/venvs/ml/     the one Python environment for this project
  ~/projects/     code
  ~/datasets/     data
  ~/models/       checkpoints / downloaded weights
  ```

## Setup

```bash
# on PARAM Shavak, after SSH'ing in:
bash setup_env.sh          # creates ~/venvs/ml, installs a CUDA-bundled fp16-safe torch stack
python test_gpu.py         # confirms CUDA visible, checks bf16/4-bit support empirically
```

If a job scheduler is present (`which sbatch`), `slurm_job.sbatch` is available as a template
— but this machine is a single workstation (not a cluster), so it most likely isn't. Default
to SSH'ing in and running commands directly.

## Ground rules when working on this box (for me and for any AI agent helping me here)

This is a shared university machine — the short version of the operating contract:

- Never touch another user's files, processes, or permissions.
- Read-only inspection first (`nvidia-smi`, `df -h`, `free -h`, `ps -u "$USER"`) before any
  action that changes state.
- No system modification, no reboot/shutdown, no service restarts, no driver/CUDA/kernel
  changes, no Docker config changes — without explicit university-admin authorization.
- No network scanning/enumeration beyond the documented VPN gateways and `10.1.19.16`.
- If a task seems to need admin privileges, say so and stop — don't try to work around it.
