# EFE-Tab — Evolutionary Feature Engineering for Tabular Classification

Evolves a `fit(X, y) → state` / `transform(X, state) → DataFrame` feature
program for tabular binary classification. The downstream model is fixed
(TabPFN); only the feature program evolves.

## How scoring works

For each candidate program the evaluator:
1. Splits the training data into Pool A (70%) and a held-out Pool B (30%)
   using a rotating seed per evaluation.
2. Runs `fit` on Pool A, then `transform` on both pools.
3. Computes a TabPFN AUC delta against the baseline (transform = identity)
   on Pool B — Pool B is **never** shown to the LLM.
4. Computes feedback artifacts on Pool A only:
   - depth-3 single decision tree (HPO-tuned) over all output features.  We used LightGBM library as it handles very basic preprocessing natively. 
   - 3-fold CV TabPFN permutation importance over the new columns
   - one-shot 3-fold CV permutation importance over the original columns
5. Reports `score = 0.5 + 5·ΔAUC − 0.10·sqrt(total_changes / n_rows) − time_penalty`,
   where `total_changes = n_new_columns + n_dropped_originals`.

The validator allows the LLM to **drop** original columns but **rejects**
in-place modification — any kept original must remain byte-identical.

## Quick start

### 1. Choose a dataset

Datasets are loaded from TabArena / OpenML and auto-downloaded on first use:
```bash
export TABARENA_TASK=churn     # or Bank_Customer_Churn, credit-g, Diabetes130US, …
```
Cached metadata lands in `~/.cache/tabarena/`. The list of supported
TabArena binary tasks is in the upstream TabArena metadata.

### 2. Set the LLM credentials

(See `../INSTALL.md` for full details.)
```bash
export AWS_BEARER_TOKEN_BEDROCK="your-ABSK-token"
```

### 3. Run an evolution loop

From the repo root:
```bash
python openevolve-run.py \
  EFE-Tab/initial_program.py \
  EFE-Tab/evaluator.py \
  --config EFE-Tab/config.yaml \
  --iterations 100 \
  --output EFE-Tab/openevolve_output
```

Checkpoints are written every 10 iterations to
`EFE-Tab/openevolve_output/checkpoints/`. The best program at any point is
in `EFE-Tab/openevolve_output/best/`.

## Useful environment variables

| Var | Default | Purpose |
|---|---|---|
| `TABARENA_TASK` | (unset) | Pick a TabArena/OpenML binary task. |
| `HOLDOUT_SEED` | 42 | Base seed for Pool A / B rotation. |
| `HOLDOUT_FRAC` | 0.3 | Pool B fraction. |
| `PERM_CV_FOLDS` | 3 | Folds for permutation importance over new columns. |
| `TREE_HPO_TRIALS` | 20 | Optuna trials for the depth-3 LightGBM tree. |
| `TREE_CV_FOLDS` | 3 | Folds for the tree HPO. |
| `ORIGINAL_PERM_CV_FOLDS` | 3 | Folds for original-column permutation importance. |
| `ORIGINAL_PERM_MAX_ROWS` | 5000 | Subsample cap for original-column importance. |

## Files

- `evaluator.py` — main evaluator (rotating Pool A/B holdout, relaxed
  validator, symmetric change penalty).
- `_evaluator_core.py` — TabPFN scoring, feature-stat / quality-check
  helpers, dataset-context builder.
- `helpers.py` — depth-3 LightGBM tree with Optuna HPO; produces the
  textual tree shown to the LLM.
- `tabarena_adapter.py` — OpenML / TabArena loader.
- `initial_program.py` — seed program (identity feature engineering).
- `config.yaml` — openevolve config (LLM, prompt, MAP-Elites database).
