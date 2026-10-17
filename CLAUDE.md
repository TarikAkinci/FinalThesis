# FinalThesis: project index

Bachelor thesis by Tarik Akinci: **demographic framing, annotator disagreement and attention steering in LLM offensiveness judgments**, on the D3CODE dataset with Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.

This file is the map for writing the thesis: where every result, script, report and paper summary lives, which numbers are current, and how the experiments were run. Paths are relative to the repo root (`~/Desktop/FinalThesis`). Most code and data sit in `D3code-main/scripts/` (abbreviated `S/` below).

Status (12 Oct 2026): all experiments done (v1, v2 = run A, v3 = runs B and C, plus follow-up analyses). Next phase: writing. Small CPU-only to-dos are in section 10.

---

## 1. Working rules (agreed with Tarik, keep following them)

- **Cluster commands: never run them, never offer to.** Tarik runs everything on COMA, LRZ and vast (ssh, rsync, sbatch, squeue). Give exact copy-paste commands and wait for pasted output. Local file edits in this repo are fine. Ask before downloads or other side effects on the laptop.
- **Equal conditions:** identical items, n, phrases and token lengths for both models and every condition. No per-model item sets, no per-variant condition subsets. If something can't be equalised (head counts, tokenisation of a fixed name), say so explicitly.
- **Verify before claiming:** check surprising results against per-item data. Test new code on a small model locally (SmolLM2-135M is cached) before cluster runs. Report confounds openly; don't oversell.
- **Statistics:** for claims about identity *content*, the phrase is the unit (phrases are identical on every item, so item-level p-values are far too optimistic). Control phrase length. For per-item correlations, control item-text length and baseline.
- **Report style:** human tone, plain language, short sentences, some bullets, **no em dashes** (check before sending). Substantive results go into a markdown file in `reports/` and are sent with SendUserFile; short questions are answered in chat.
- **rsync narrowly** (specific folders, never the whole tree). Files over 100 MB never go into git. Large transfers from COMA: split into 300 MB pieces (section 9).
- Commit or push only when asked. `.npz` files and `results/v2/npz_parts/` are gitignored on purpose.

## 2. Research questions and the one-paragraph story

- **RQ1:** Do LLMs change their offensiveness judgments when given a demographic identity, beyond what content-free phrases do?
  - Yes for region; in Qwen's perspective formats mostly via the place name. No for gender and religion.
- **RQ2:** Do these changes match how real members of those groups judge the texts?
  - No. The models over-flag, and framing moves them further from every group's real ratings.
  - Moreover, D3CODE groups barely differ text by text; they differ only by small average regional strictness offsets. Llama gets the order of these offsets right but exaggerates them.
- **RQ3:** Where does the identity effect arise in the model, and can attention steering control it?
  - A handful of middle-layer heads (Llama 8 to 13, Qwen 13 to 17) carry it causally. The information reaches the answer via intermediate tokens.
  - PASTA-style steering can remove or amplify it. Llama's "dampener" heads remove the identity bias specifically.
  - Steering's gain in "closeness to real raters" is calibration: a constant shift does as well.

Story in one line: **demographic prompting injects a model-made group effect instead of recovering real group perspectives; attention steering can locate and switch off that effect, but cannot create alignment where the data has almost no group-specific signal.**

## 3. Repository map

| path | what it is |
|---|---|
| `CLAUDE.md` | this index |
| `reports/` | every write-up: results reports, plans, literature summaries, runbooks, proposal (section 7) |
| `paper-summaries.xlsx` | Tarik's own paper tracking sheet |
| `D3code-main/dataset/` | `d3-items.csv` (4,590 texts), `d3-ratings.csv` (~145k ratings), `d3-raters.csv` (4,309 raters: age, gender, region, country, SES, 6 moral foundations), `mhs.csv` (Measuring Hate Speech, early zero-shot only) |
| `D3code-main/D3 Data Card.pdf`, `README.md`, `splits.json` | dataset documentation; frozen train/val/test split (3164/678/678), only used by the early zero-shot runs |
| `D3code-main/plots/`, `plots/` | phase-1 data exploration figures (rating distributions, raters per region, MFQ by region, disagreement by category) |
| `S/` = `D3code-main/scripts/` | all experiment and analysis code (section 5) |
| `S/items/items_v2.csv` | **the frozen v2/v3 item list**: 500 texts, split A/B, category, entropy, rater share, baselines of both models |
| `S/results/` | all numeric outputs (section 6) |
| `S/figures/` | figures produced by scripts (section 8) |
| `S/logs/` | job logs per phase (`v2/`, `v3/`, `master_grid/` = LRZ v1, older ones) |
| `S/vast/` | `setup_vast.sh`, `run_pipeline.sh` (stage runner, also used inside the COMA jobs) |
| `S/archive/` | dead early scripts (raw probing, activation extraction, an early head-steering attempt). History only. |
| `_cloud_bundle.tgz` | old code bundle, not needed |

## 4. Timeline of the work (phases, what each established, where it lives)

**Phase 0: data exploration.** `reports/analysis_report.md`, `D3code-main/plots/`.

**Phase 1: zero-shot baseline (seeded 500-item eval sample of the test split, generate() + parsing).**
- `S/zeroshot_two_datasets.py`, `S/evaluate.py`, `S/bootstrap_ci.py`.
- Outputs: `S/results/zeroshot/results_zeroshot_tuned.csv`, per-item predictions. Report: `reports/zeroshot_results.md`.
- Both models over-flag on D3CODE and MHS. Llama D3CODE: F1 0.39, predicts 83% positive vs 20% true, AUC 0.83; tuned threshold 0.99 gives macro-F1 0.65.
- **Superseded for the thesis by the full-dataset numbers (section 4, v2 scores).** Keep MHS only as an early side result if wanted.

**Phase 2: v1 pilots on the seeded sample (Llama, later Qwen).** History; their numbers are superseded by v2. Keep as "pilot studies" that motivated the v2 design.
- Attention probing (30 items, 7 conditions): `S/attention_probe_detailed.py`, `S/analyze_attention_*.py`. Results: `S/results/attention_probing/`. Reports: `reports/attention_probing_final_report.md`, `reports/attention_layer_analysis.md`. Findings: about 80% of attention goes to the sink; content attention is about 87% instruction and 13% item text; the demographic effect sits in layers 9 to 18, peak 13.
- Disagreement case studies and placebo studies (6, 100 and 500 items): `S/attention_disagreement_case_studies.py`, `S/attention_disagreement_scaled.py`, `S/repair_disagreement_scaled_results.py`, `S/select_items_with_baseline_filter.py`. Reports: `reports/disagreement_*.md`. These established the **ceiling-effect confound** (which led to the baseline band filter) and the odd placebo direction (which led to the placebo pool).
- Prompt variants (prefix, suffix, reworded, embedded): `S/attention_prompt_variants.py`, `S/results/prompt_variants/`.

**Phase 3: v1 master grid on LRZ (block design, per-model in-band items: Llama 236, Qwen 64).**
- `S/master_grid.py` (old version in git history), `S/build_master_table.py`.
- Results: `S/results/master_grid/`, `S/results/master_table.*`, `S/results/analysis_lrz/`. Report: `reports/master_table_full_analysis.md`.
- **Superseded by v2.** Problems that v2 fixed: unequal item sets; bf16-rounded P(yes); a span bug (identity found inside the item text); the target variant double-counting one token; a placebo in the context variant containing "offensive".

**Phase 4: v2 = run A (COMA H200, 7 Oct 2026).** The main behavioural dataset.
- **Items:** `S/select_items.py` scored all 4,519 unique D3CODE texts with both models (`S/results/v2/scores/`). 554 are in band [0.02, 0.98] for both; 500 drawn with category quotas (153 social-group, i.e. all available; 275 random; 72 moral); A/B split 250/250.
- **Grid:** full factorial, 22 conditions × 10 variants × 500 items per model (110,500 rows), with log-odds, fp32 decision logits, raw and value-weighted per-head attention.
- **Checks:** `S/check_matching.py`, `S/check_reproduction.py` (v1 vs v2: rank rho ≥ 0.995, item-level differences about 0.1 log-odds from bf16 rounding).
- Outputs: `S/results/v2/`. Reports: `reports/v2_behavioural_analysis.md`, `reports/supervisor_summary_v2.md`. Runbooks: `reports/coma_runbook_v2.md` (`reports/vast_runbook_v2.md` was prepared but not used).

**Phase 5: v3 step 1, per-head attention analysis (CPU, split A).** `S/analysis_perhead_v3.py`. Outputs: `S/results/v2/analysis/perhead_v3_*`. Explained from scratch in `reports/v3_step1_explained.md`.

**Phase 6: v3 run B (COMA).** Report: `reports/v3_runB_results.md`.
- B1, place-name controls ("someone researching / mapping <region>"): `S/v3_place_job.sbatch`. Output: `S/results/v3/place/`.
- B2, PASTA head profiling, one head at a time on split A: `S/head_profiling.py`, `S/v3_profile_job.sbatch`. Output: `S/results/v3/profiling/`. Analysis: `S/analysis_profiling_v3.py`. Frozen head sets: `S/head_selection_v3.py` writes `S/results/v3/analysis/heads_*`.

**Phase 7: v3 run C, steering (COMA, split B).**
- `S/steer_eval.py`, `S/v3_steer_job.sbatch`, upper bounds via `S/ablation_heads.py`.
- Analysis: `S/analysis_steering.py`. Figure: `S/fig_steering_v3.py`.
- Outputs: `S/results/v3/steering/`, `S/results/v3/ablation/`, `S/results/v3/analysis/steering_*`. Report: `reports/v3_runC_results.md`.

**Phase 8: follow-ups (CPU).** Is the model better without a demographic? Is there a group signal at all (demographic and moral)? Standard metrics. Report: `reports/v3_identity_vs_baseline.md`; table `S/results/v3/analysis/q_identity_vs_unframed.csv`.

## 5. Script index (S/ = D3code-main/scripts/)

**Shared modules**

| file | role |
|---|---|
| `master_grid.py` | the v2/v3 core: conditions, placebos, place controls, prompt variants, span finding (`locate_spans`, `spans_problem`), preflight, `DecisionHead` (fp32 decision logits), `ValueNorms` (value-weighted attention), the grid run. Env vars: `MODEL_NAME`, `ITEMS_CSV`, `RESULTS_DIR`, `MAX_ITEMS`, `PREFLIGHT_ONLY`, `ONLY_CONDITIONS` + `OUT_SUFFIX`, `NOISE_ITEMS` |
| `span_utils.py` | token-span finding; `last=True` for spans after the item text |
| `analysis_utils.py` | loaders (`load_grid`, `load_perhead`), BH-FDR, vectorised Spearman, paired tests |
| `evaluate.py` | shared metrics (rating distribution, entropy) |
| `zeroshot_two_datasets.py` | the canonical `PROMPT`, the early zero-shot runs |
| `high_disagreement_items.py` | the old seeded 500-item sample (`rank_by_entropy`) |

**v2/v3 pipeline (live)**

| file | role | runs on |
|---|---|---|
| `select_items.py` | `score` (all D3CODE texts per model) and `select` (shared list, N_TARGET, A/B split) | GPU / CPU |
| `check_matching.py` | identical span lengths across both tokenizers | laptop |
| `check_reproduction.py` | v2 vs v1 on identical prompts | CPU |
| `gpu_check.py` | checks bf16 kernels work, prints versions | GPU |
| `analysis_v2_behaviour.py` | Phase-2 behavioural tests → `results/v2/analysis/behaviour_*.csv` | CPU |
| `analysis_group_gap.py` | relative group alignment (across items, within item, per head) | CPU |
| `analysis_perhead_v3.py` | step-1 per-head content and behaviour tests, frozen candidate rule | CPU |
| `head_profiling.py` | PASTA single-head profiling; `--span identity` or `--span text` | GPU |
| `analysis_profiling_v3.py` | profiling reliability, nulls, overlap with step 1 | CPU |
| `head_selection_v3.py` | freezes dampener and amplifier sets for run C | CPU |
| `ablation_heads.py` | attention hook (`set_intervention`), ablation and amplification, random head sets, upper bounds | GPU |
| `steer_eval.py` | run C: attention steering (identity or text) and activation steering (actadd) | GPU |
| `analysis_steering.py` | success criterion: distance to real raters, specificity, comparison with random heads | CPU |
| `fig_steering_v3.py` | dose-response figure | CPU |
| `v2_{select,smoke,grid}_job.sbatch` | COMA jobs for run A | COMA |
| `v3_{place,profile,steer}_job.sbatch` | COMA jobs for runs B and C | COMA |
| `vast/run_pipeline.sh` | stage runner (check, score, select, preflight, smoke, grid, reproduce, bundle, profile, ablation) | any |

**Legacy (v1 and pilots):** `attention_probe*.py`, `analyze_attention_*.py`, `attention_disagreement_*.py`, `attention_prompt_variants.py`, `repair_disagreement_scaled_results.py`, `select_items_with_baseline_filter.py`, `build_master_table.py`, `analysis_length_control.py`, `analysis_sink_dynamics.py`, `analysis_perhead_candidates.py`, `analysis_perhead_compare.py`, `master_grid*_lrz.sbatch`, `master_grid_job.sbatch`, `submit_master_grid_chain.sh`, `baseline.py`, `create_splits.py`.

## 6. Results index

**`S/results/v2/` (run A)**
- `scores/scores_<MODEL>.csv`: baseline P(yes) and log-odds for all 4,519 texts. Source of the **full-dataset standard metrics**.
- `master_grid/master_grid_<MODEL>.csv.gz`: one row per item × variant × condition (and 500 baseline rows). Columns include `delta_logodds_yes`, `prob_yes`, `split`, `category`, `entropy`, `p_offensive_raters`, `matched_placebos`, `demo_span_tokens`, and attention shares raw and `_vw`.
- `master_grid/master_grid_perhead_<MODEL>.npz` (1.6 / 1.2 GB, not in git): `attention`, `attention_vw` (rows × layers × heads × 4 segments: demographic, instruction, item text, other) plus index arrays.
- `master_grid/matching_*`, `noise_*`: span-length table, repeat-run noise (exactly 0).
- `reproduction_check.csv`: v1 vs v2.
- `analysis/behaviour_*.csv`: conditions, identity_specific (exact-length, all-placebo and length-line nulls), axis_tests (incl. length-adjusted), paraphrase consistency, moderators, group_alignment, cross_model.
- `analysis/group_gap_*.csv`: relative alignment.
- `analysis/perhead_v3_*`: step-1 per-head tables, candidates, `perhead_v3_rule.txt`, summary.
- `analysis/v3_region_vs_place.csv`: region vs place-name controls (behavioural).
- `npz_parts/`: transfer pieces, can be deleted.

**`S/results/v3/` (runs B and C, follow-ups)**
- `place/`: place-control grid (CSV + per-head npz, suffix `_place`).
- `profiling/profiling_<MODEL>_<variant>.csv.gz`: every head × item × condition change. `profiling_heads_*`: per-head summary. `profiling_text_*`: text-span profiling for arm D.
- `steering/steering_<MODEL>_<label>.csv.gz`: labels `attention_identity_dampeners`, `attention_identity_amplifiers`, `attention_text`, `actadd`. Columns: `head_set` (selected, random_k, none), `strength`, `variant`, `condition`, `logodds_yes`.
- `ablation/`: upper bounds (`all_heads_last`, `all_heads_all`).
- `analysis/`:
  - `profiling_heads.csv`, `profiling_summary.csv`, `profiling_null_*`, `profiling_overlap.csv`;
  - `heads_<MODEL>_{dampeners,amplifiers}.csv`, `heads_rule.txt`;
  - `steering_summary.csv`, `steering_vs_trivial.csv`, `steering_plus_shift_cv.csv`;
  - `q_identity_vs_unframed.csv`.

**Older:** `S/results/zeroshot/`, `attention_probing/`, `disagreement_*`, `prompt_variants/`, `master_grid/` (v1 LRZ), `analysis_lrz/`, `master_table*`.

## 7. Reports index (`reports/`)

**Current results (cite these)**

| file | content |
|---|---|
| `v2_behavioural_analysis.md` | Phase 2: all behavioural results of run A |
| `v3_step1_explained.md` | attention, heads and steering explained from scratch, plus step-1 results (good basis for the background and method chapters) |
| `v3_runB_results.md` | place-name controls and causal head profiling, frozen heads |
| `v3_runC_results.md` | steering results, success criterion, constant-shift comparison, upper bounds |
| `v3_identity_vs_baseline.md` | unframed vs framed vs steered after calibration; group-signal reliability (demographic and moral); regional offsets; standard metrics |
| `supervisor_summary_v2.md` | one-page summary for the supervisor |
| `v3_plan.md` | v3 plan, decisions, supervisor-confirmed direction and success criterion |
| `metrics_glossary.md` | metric definitions (written for v1; extend with v2/v3 metrics, see to-dos) |

**Literature**

| file | content |
|---|---|
| `Final Thesis Cross-Cultural Disagreement in Offens ... .md` | Notion export: the main literature list with summaries (persona effect, LLM overconfidence on disagreement, cross-cultural hate speech, perspectivism, D3CODE, GRASP, jury learning, HateXplain disagreement, moral correlates, demographic prompting at scale, ...) |
| `key_points_thesis_proposal.md` | key literature points (e.g. demographics explain only 1 to 10% of annotation variance) |
| `technical_papers_sum.md` | 6 technical papers: political views and steering, FairSteer, steering toward fairness, probing ideology, **PASTA**, TDD |
| `attention_probing_papers_summary.md` | the same papers framed for the supervisor's questions |
| `paper_summaries.md`, `literature_comparison_report.md` | papers compared with the v1 findings (persona bias, recency bias, Muslim bias, ...) |
| `paper-summaries.xlsx` (repo root) | Tarik's tracking sheet |

**Proposal and planning:** `thesis_proposal.md/.pdf` (Tarik's final proposal), `proposal_sketch.md`, `next_steps.md`, `llm_experiment_practical_guide.md`.

**Runbooks:** `coma_runbook_v2.md`, `vast_runbook_v2.md`.

**Superseded (history, pilot studies):** `zeroshot_results.md`, `attention_probing_final_report.md`, `attention_layer_analysis.md`, `disagreement_case_studies_report.md`, `disagreement_final_report.md`, `disagreement_scaled_report.md`, `disagreement_v3_report.md`, `master_table_full_analysis.md`, `analysis_report.md` (data exploration, still useful for the data chapter).

## 8. Figures

- `S/figures/v3/steering_dose_response.png`: distance to real raters vs steering strength (dampeners, text, random heads, constant-shift line). Main run C figure.
- `S/figures/v2/content_attention_charts/`: token-level content attention, prefix / infix / suffix, 3 illustrative items, both models.
- `D3code-main/plots/`, `plots/`: data exploration (rating distributions, raters, MFQ by region, disagreement by category).
- Older: `S/figures/attention_probing`, `disagreement_*`, `master_grid`, `prompt_variants`.
- Still to make for the thesis: section 10.

## 9. Key numbers (with source)

**Data (D3CODE).** 4,590 texts (4,519 unique with ≥ 10 raters), 4,309 raters, about 145k ratings, 8 regions. Mean rating 1.13 / 4; 32% of ratings "offensive"; 21.7% of texts majority-offensive. Our 500 v2 texts: offensive share 0.21 vs 0.34 overall, entropy 1.49 vs 1.77, 13 majority-offensive. (`S/results/v2/scores`, the select log, `reports/v3_identity_vs_baseline.md`)

**Standard metrics, unframed prompt, all 4,519 texts** (`v3_identity_vs_baseline.md` §5):

| | Llama | Qwen |
|---|---|---|
| texts flagged | 78.7% | 85.0% |
| precision | 0.27 | 0.25 |
| recall | 0.99 | 0.99 |
| F1 | 0.43 | 0.40 |
| macro-F1 | 0.42 | 0.36 |
| AUC | 0.83 | 0.82 |
| Spearman vs share | 0.66 | 0.67 |

**v2 behaviour** (`v2_behavioural_analysis.md`)
- Region vs placebos, length-adjusted, prefix: Llama +1.62 log-odds (p < 0.001), Qwen +2.78 (p = 0.001). Significant in all 10 variants (p ≤ 0.006).
- Gender, religion and bare identity are not distinguishable from same-length placebos (prefix: Llama +0.22 / -0.10 / +0.70; Qwen +0.27 / +1.17 / +0.96, all n.s.). Exception: Llama context variants, gender +1.07 (p = 0.007).
- Placebos: negative length slope in every variant. Qwen placebos push toward "offensive" by +2.3 to +3.8. The tea placebo is an outlier in embedded and target.
- Paraphrase templates agree (rho 0.83 to 0.99). Suffix pulls everything down; in Qwen it reverses the identity effect (-2.2). Target measures topic relevance (placebos collapse).
- Flips under prefix: Llama 29% vs 21% (demographic vs placebo); Qwen 38% vs 34%.
- Group distance: Llama 0.35 → 0.51 framed, Qwen 0.40 → 0.68. 0 of 240 relative-alignment tests significant. Rater entropy doesn't predict sensitivity. Cross-model condition means rho 0.67.

**Place-name controls** (`v3_runB_results.md` §1): region minus "researching / mapping" the same region. Llama prefix +0.73 (p = 0.013), infix +1.01 (p = 0.003). Qwen prefix +0.02 (n.s.), infix +1.17, target +5.09. So Llama's region effect is about half identity; Qwen's perspective-format effect is the place name.

**Per-head step 1** (`v3_step1_explained.md`): sink heads 302 / 1,024 (Llama), 104 / 784 (Qwen). Candidates (raw attention) 34 and 108; Llama peak layer 14. Item-text length confound fixed by partial correlation.

**Profiling** (`v3_runB_results.md` §2): split-half reliability 0.82 to 0.96. Heads beyond both nulls (placebo-pair and place-pair, 99th percentile):

| | prefix | infix | target |
|---|---|---|---|
| Llama | 7 | 2 | 1 |
| Qwen | 4 | 6 | 3 |

- Location: Llama layers 8 to 13, Qwen 13 to 17.
- Dampeners: Llama 12.8, 12.6, 11.23, 8.26, 11.25; Qwen 14.10, 13.24, 6.0.
- Amplifiers: Llama 13.30, 13.4; Qwen 14.24, 13.3, 13.14, 17.18.
- Overlap with the step-1 correlational candidates is barely above chance.

**Steering** (`v3_runC_results.md`). Distance to the real group, prefix, split B:

| | Llama | Qwen |
|---|---|---|
| unsteered | 0.50 | 0.67 |
| dampeners at strength 100 | 0.22 | 0.43 |
| text steering at strength 100 | 0.20 | 0.20 |
| random heads | 0.50 | 0.66 |
| activation steering, framing direction removed | 0.38 | 0.66 |
| **constant shift** | **0.19** | **0.22** |

- Steering on top of a fitted shift gains at most 0.02.
- Text steering flags only 0 to 3% of texts and lowers ranking ability.
- Llama dampeners: regions -2.5 log-odds, placebos +1.7, place names +0.6; ranking 0.15 → 0.18.
- Relative alignment under steering: rho -0.04 to +0.03.
- Upper bounds: removing identity attention at the answer position only removes 5 to 19% of the effect (Llama), about 0% (Qwen); at all positions 89 to 98% (Llama), 94% (Qwen prefix).

**Group signal** (`v3_identity_vs_baseline.md`)
- Per-text group gaps don't replicate across disjoint rater halves: gender 0.01 / 0.00, regions 0.00 to 0.05 (Sinosphere and Sub-Saharan Africa about 0.1), random pseudo-groups 0.02.
- Average regional offsets: Arab Culture +6.6 points, Latin America +5.6, Oceania -6.5, Sinosphere -5.6 (p = 1e-37). Gender ±0.5 (n.s.).
- Moral foundations: same pattern (split-half 0.00 to 0.07; offsets up to +5.6 for high care).
- Within-region entropy 0.64 vs 0.80 overall.
- Calibrated distances: unframed 0.19 / 0.20, framed 0.19 / 0.24. Predicting each group with the all-raters share: 0.10.
- Llama's region order matches the real offsets (rho 0.81, n = 8) but with twice the spread; Qwen 0.05.

## 10. Thesis writing plan and open to-dos

**Proposed outline:**
1. Introduction and RQs
2. Background: LLMs for offensiveness detection, annotator disagreement and perspectivism, persona prompting, attention, interpretability and steering (PASTA, activation steering)
3. Data and setup: D3CODE, item selection, prompt variants, placebos and controls, metrics
4. Results I, behaviour (v2 + place controls)
5. Results II, mechanism (per-head, profiling, information flow)
6. Results III, steering (run C)
7. Results IV, is there a group signal? (reliability, offsets, moral foundations)
8. Discussion: calibration vs alignment, implications for moderation, limitations
9. Conclusion

The v1 work goes in as pilot studies (methods chapter or appendix) that motivated the v2 design: ceiling effect → band filter; placebo direction → placebo pool; length → exact token matching; unequal item sets → shared item list.

**To-dos (all CPU, no GPU needed):**
- Standard-metrics table for every condition and steering setting: F1, precision, recall, AUC, plus a soft-label score (cross-entropy or Brier) vs the raters' share.
- Recompute the pilot-era descriptives from v2 data: sink share, instruction vs text share, layer profile (from the v2 npz); confidence vs rater disagreement on all 4,519 texts; threshold tuning on the full scores (tune on one half, evaluate on the other).
- Thesis-ready figures for chapters 4 to 7 (use the dataviz skill; validate the palette).
- Extend `metrics_glossary.md` with the v2/v3 metrics (length-adjusted null, phrase-level p, value-weighted attention, profiling nulls, distance to raters, constant shift).
- Methods text: the exact prompt per variant (`S/master_grid.py: build_content`), selection, matching, statistical units.
- Optional: re-check the step-1 correlational candidates against the place controls (`S/results/v3/place/*_place.npz` is present locally; verify integrity first).
- Optional GPU: the demographic framings on all 4,519 texts (prefix, no attention) for per-condition F1, about 30 min per model on COMA.
- Decide whether MHS stays in the thesis (it is only in the early zero-shot run).

## 11. Experimental design details (for the methods chapter)

- **Base prompt** (`S/zeroshot_two_datasets.py`): `Text: "{text}"\nIs this text offensive? Answer with only 0 (no) or 1 (yes).` Chat template of each model. Single forward pass; P(yes) = softmax over the "0" and "1" logits, computed in fp32 from the final hidden state (`DecisionHead`).
- **Variants** (`build_content`), identity sentence or noun in brackets:
  - prefix `[You are X.] <base>`;
  - reworded `[Assume you are X.]`, imagine `[Imagine you are X.]`, consider `[Consider that you are X.]`;
  - infix `Text: "..."\n[You are X.] Is this text offensive? ...`;
  - suffix `<base> [You are X.]`;
  - embedded `... As [X], is this text offensive? ...`;
  - target `... Do you think this text is offensive for [X]? ...`;
  - context `[You are X.] Your background and everyday experience shape how you read things. <base>`;
  - context_long `... + Answer from that perspective.`
- **Conditions:**
  - 14 demographics: 8 D3CODE regions as "someone from <Region>", a man / a woman, a Christian / a Muslim, an Arab / a European;
  - 8 placebos: a runner, a cyclist, an early riser, someone who plays chess, someone who collects old maps, someone who enjoys crossword puzzles, someone who usually takes the stairs, someone who prefers tea over coffee. Two per noun length (2, 4, 5, 6 tokens), identical lengths under both tokenizers in all variants;
  - 8 place controls (v3): someone researching / mapping <4 regions>.
- **v3 regions:** Sub-Saharan Africa, Arab Culture, Western Europe, Latin America.
- **Item selection:** baseline P(yes) in [0.02, 0.98] for both models; N = 500; category quotas proportional to the pool, capped by availability; split A (selection) and B (held-out) of 250 each, stratified by category.
- **Steering:** PASTA-style, in the chosen heads at every query position: attention to the span × strength, row renormalised (strength 100 = 0.01 on everything else). Random-head controls have the same per-layer counts. Activation steering: add or subtract the mean framed-minus-unframed hidden state at the middle layer, answer position.
- **Success criterion (supervisor, 10 Oct):** smaller mean |P(yes) - share of the group's raters| on split B (texts with ≥ 4 group raters); all raters for unframed prompts and arm D.

## 12. Clusters and running jobs

- **COMA (current):** `tarik.akinci@login.coma-cluster.de`. Mirror at `~/FinalThesis/D3code-main/scripts`. Env `~/thesis-env-v2`: uv Python 3.12, torch 2.14.1+cu130, transformers 5.14.1. GPUs: H200. Partition `compute`, `--gres=gpu:nvidia:1`. `logs/v2` and `logs/v3` must exist before sbatch. The old `~/thesis-env` (Python 3.9) is v1 only.
- **Sync home, large files:** on COMA `split -b 300M -d <file> ~/parts/<file>.part.`; on the laptop `rsync -av --partial -e "ssh -o ServerAliveInterval=15" ...`, then `cat` the pieces and verify (`zipfile.testzip()` for npz, or sha256). Never auto-retry in a tight loop (the server blocks repeated logins).
- **LRZ AI Systems:** used for v1 only (`master_grid_*_lrz.sbatch`). The V100 nodes can't run torch ≥ 2.13 CUDA 13 builds.
- **vast.ai:** `S/vast/` prepared but not used.

## 13. Gotchas learned the hard way

- bf16 output logits quantise P(yes) (log-odds steps of 0.06 to 0.125). Always use `DecisionHead` (fp32 decision logits).
- transformers 5 loads the checkpoint dtype by default. Pass `dtype` explicitly in tests.
- Spans placed after the item text must be searched from the end (item texts contain "a man", "a Muslim", ...). `spans_problem` rejects overlapping or misplaced spans.
- Phrase length drives attention and behaviour. Use per-token attention and length-adjusted nulls. Item-text length confounds per-item attention-behaviour correlations; use partial correlation.
- Batched GPU runs differ from single runs by up to about 0.5 log-odds (bf16 and padding). Compare steered vs unsteered within the same batch.
- Single-head steering at strength 100 changes almost everything. Judge heads against placebo-pair and place-pair nulls, not item-level p-values.
- `--coefs=-1,...` needs the `=` (a negative number otherwise reads as a flag).
- Group gaps from 2 to 7 raters are noise. Relative-alignment nulls say more about the data than about the model.
- Steering that lowers P(yes) always "improves" distance to raters because the models over-flag. Always compare with a constant shift and check ranking and the flag rate.
