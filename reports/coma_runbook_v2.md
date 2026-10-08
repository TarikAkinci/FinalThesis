# Run A on COMA: step by step

Same experiment as the vast.ai plan: score and select the items, smoke test, then the full v2 grid for Llama and Qwen. You run every command yourself and paste me the output at the marked steps.

Host: `tarik.akinci@login.coma-cluster.de`. Everything below assumes the cluster mirror lives at `~/FinalThesis/D3code-main/scripts`, as before.

## 1. Look around on the login node -> paste me the output

```bash
ssh tarik.akinci@login.coma-cluster.de
```

Then, on the cluster:

```bash
python3 --version; ls /usr/bin/python3.* 2>/dev/null; which uv conda 2>/dev/null; module avail 2>&1 | grep -i python
```

```bash
sinfo -o "%P %N %G %t"
```

```bash
ls ~/.cache/huggingface/hub; df -h ~
```

What I need from this: which Python 3.10+ exists (the new code needs it; the old `~/thesis-env` is Python 3.9), which GPUs are up, and whether both models are still cached.

## 2. New Python environment (on the cluster)

COMA only has Python 3.9, no conda and no modules. `uv` installs into your home directory with the old pip and brings its own Python 3.12 (same version as the laptop). The old `~/thesis-env` stays untouched.

```bash
python3 -m pip install --user uv
```

```bash
~/.local/bin/uv venv --seed --python 3.12 ~/thesis-env-v2
```

```bash
source ~/thesis-env-v2/bin/activate && pip install --upgrade pip && pip install torch "transformers==5.14.1" "huggingface_hub>=1.0" pandas scipy statsmodels matplotlib
```

```bash
python -c "import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__)"
```

## 3. Copy the code over (on the laptop)

```bash
rsync -av ~/Desktop/FinalThesis/D3code-main/scripts/{master_grid,span_utils,select_items,check_matching,check_reproduction,head_profiling,ablation_heads,analysis_utils,zeroshot_two_datasets,evaluate,gpu_check}.py ~/Desktop/FinalThesis/D3code-main/scripts/v2_{select,smoke,grid}_job.sbatch tarik.akinci@login.coma-cluster.de:FinalThesis/D3code-main/scripts/
```

```bash
rsync -av ~/Desktop/FinalThesis/D3code-main/scripts/vast tarik.akinci@login.coma-cluster.de:FinalThesis/D3code-main/scripts/
```

Then on the cluster, the log folder (without it the jobs vanish with no log):

```bash
mkdir -p ~/FinalThesis/D3code-main/scripts/logs/v2
```

## 4. Score and select (GPU, about 10 minutes) -> paste me the output

```bash
cd ~/FinalThesis/D3code-main/scripts && sbatch v2_select_job.sbatch
```

When it finished (`squeue -u tarik.akinci` no longer lists it):

```bash
tail -45 ~/FinalThesis/D3code-main/scripts/logs/v2/slurm_select_<JOBID>.log
```

The end shows how many items are in band for both models and the final N. If N is clearly below 400, we decide together before going on.

## 5. Preflight and smoke test (GPU, about 10 minutes) -> paste me the output

```bash
cd ~/FinalThesis/D3code-main/scripts && sbatch v2_smoke_job.sbatch
```

```bash
grep -E "GPU OK|Preflight|rows |NaN|span length varies|identical input|passes/s|Error|Traceback" ~/FinalThesis/D3code-main/scripts/logs/v2/slurm_smoke_<JOBID>.log
```

## 6. Full grid (GPU, about 2 to 4 hours, both models in one job)

One job for both models, so both run on the same GPU.

```bash
cd ~/FinalThesis/D3code-main/scripts && sbatch v2_grid_job.sbatch
```

Progress: `tail -3 ~/FinalThesis/D3code-main/scripts/logs/v2/slurm_grid_<JOBID>.log`

## 7. Copy the results home (on the laptop)

```bash
rsync -av tarik.akinci@login.coma-cluster.de:FinalThesis/D3code-main/scripts/results/v2 ~/Desktop/FinalThesis/D3code-main/scripts/results/
```

```bash
rsync -av tarik.akinci@login.coma-cluster.de:FinalThesis/D3code-main/scripts/logs/v2 ~/Desktop/FinalThesis/D3code-main/scripts/logs/
```

```bash
rsync -av tarik.akinci@login.coma-cluster.de:FinalThesis/D3code-main/scripts/figures/v2 ~/Desktop/FinalThesis/D3code-main/scripts/figures/
```

```bash
rsync -av tarik.akinci@login.coma-cluster.de:FinalThesis/D3code-main/scripts/items ~/Desktop/FinalThesis/D3code-main/scripts/
```

Tell me when it is synced. I then run the reproduction check against the LRZ results locally, and check every output.

Keep a second copy of `results/v2/master_grid/*.npz` (cloud drive or external disk). They are not in git.
