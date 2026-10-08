# Run A on vast.ai: step by step

Run A = item scoring and selection, then the full v2 grid for Llama and Qwen. Expected time is about 2 to 2.5 hours on one RTX 4090 (roughly $1 to $1.50). You run every command yourself. After steps 5 and 6 you paste me the output before going on.

## 0. On the laptop: push the new code

The instance clones from GitHub, so the new scripts have to be on GitHub first.

```bash
cd ~/Desktop/FinalThesis
git add .gitignore D3code-main/scripts/master_grid.py D3code-main/scripts/span_utils.py D3code-main/scripts/select_items.py D3code-main/scripts/check_matching.py D3code-main/scripts/check_reproduction.py D3code-main/scripts/head_profiling.py D3code-main/scripts/ablation_heads.py D3code-main/scripts/analysis_utils.py D3code-main/scripts/vast reports/vast_runbook_v2.md
git commit -m "v2 design: shared items, full factorial, exact placebo matching, run A/B pipeline"
git push
```

## 1. Rent the instance

- 1x RTX 4090, PyTorch template
- disk 100 GB (models take about 31 GB, results 3 to 5 GB)
- RAM 32 GB or more (the per-head arrays are held in memory until the end of each model)
- high download speed (the models are about 31 GB)

## 2. Connect and start tmux

Use the SSH command vast shows you (it looks like `ssh -p PORT root@IP`). Then:

```bash
tmux new -s grid
```

If the connection drops, reconnect and run `tmux attach -t grid`. To leave it running on purpose, press Ctrl-b, then d.

## 3. Code, login, setup

```bash
cd /workspace && git clone https://github.com/TarikAkinci/FinalThesis.git && cd FinalThesis
```

```bash
hf auth login
```

(paste your Hugging Face token yourself; it needs Llama 3.1 access)

```bash
bash D3code-main/scripts/vast/setup_vast.sh
```

This checks the GPU really runs, installs the pinned libraries and downloads both models. It ends with "Setup done".

## 4. Go to the scripts folder

```bash
cd /workspace/FinalThesis/D3code-main/scripts
```

## 5. Score and select items (about 10 minutes) -> paste me the output

```bash
bash vast/run_pipeline.sh check score select
```

The important part is the end of `select`: how many items are in band for both models, the final N, the A/B split per category, and the selection-bias lines. If N is clearly lower than 400, we decide together before going on.

## 6. Preflight and smoke test (about 5 minutes) -> paste me the output

```bash
bash vast/run_pipeline.sh preflight smoke
```

Check for "Preflight OK" twice, the sanity-check block of both smoke runs, and the speed line (passes/s), which gives the real time estimate for step 7.

## 7. Full grid, reproduction check, bundle (about 2 hours)

```bash
bash vast/run_pipeline.sh grid reproduce bundle
```

You can detach (Ctrl-b, then d) and close the laptop. The last line prints the copy command.

## 8. On the laptop: copy everything home

Replace PORT and IP with the values from the vast SSH command:

```bash
scp -P PORT root@IP:/root/bundle_v2.tgz ~/Desktop/FinalThesis/D3code-main/scripts/
```

```bash
cd ~/Desktop/FinalThesis/D3code-main/scripts && tar xzf bundle_v2.tgz && ls results/v2/master_grid
```

You should see two `master_grid_*.csv.gz`, two `master_grid_perhead_*.npz`, two `matching_*.csv` and two `noise_*.csv`.

## 9. Before destroying the instance

- Make a second copy of `bundle_v2.tgz` (cloud drive or external disk). The `.npz` files are not in git and disappear with the instance.
- Only destroy the instance once step 8 worked and the files open locally. Tell me and I check them.
