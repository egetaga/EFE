# EFE-Time — Evolutionary Time-Series Transform Discovery

Evolves an invertible time-series transform — a `TransformProgram` with
`fit(y_hist, meta)`, `transform(y, state)`, and `inverse_transform(z, state)` —
that is applied before a fixed Chronos-2 forecaster. The forecaster does
not change, only the transform evolves.

## How scoring works

A 2-stage cascade evaluator:

**Stage 1 — validity gate (no forecasting).** Checks that the program
- preserves length, order, and NaN positions,
- returns finite values,
- has a self-consistent inverse (`|inverse(transform(y)) - y| < 1e-4`),
- handles forecast-length arrays in `inverse_transform`.

**Stage 2 — Chronos-2 MASE evaluation.** For each evaluation series and each
rolling-origin window:
1. Fit on the history; transform; feed the transformed history to Chronos-2;
   inverse-transform the forecast back to the original scale.
2. Score the candidate's MASE against an identity-transform baseline that
   feeds the raw history to the same Chronos-2 model.

The score is `0.5 + 5·(1 − candidate_MASE/baseline_MASE).
This `config.yaml` has population_size: 60 with 3 islands. 

## Quick start

### 1. Get a GIFT-Eval dataset

```bash
pip install "git+https://github.com/SalesforceAIResearch/gift-eval.git"

huggingface-cli download Salesforce/GiftEval \
  --repo-type=dataset \
  --local-dir /path/to/gift_eval_data \
  --include "m4_yearly/*"        # or any other GIFT-Eval dataset directory

export GIFT_EVAL=/path/to/gift_eval_data
export TS_DATASET=covid_deaths      # or m4_yearly etc. …
```

### 2. Set LLM credentials

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-ABSK-token"
```

### 3. Run an evolution loop

```bash
python openevolve-run.py \
  EFE-Time/initial_program.py \
  EFE-Time/evaluator.py \
  --config EFE-Time/config.yaml \
  --iterations 100 \
  --output EFE-Time/openevolve_output
```

Checkpoints land in `EFE-Time/openevolve_output/checkpoints/`.

## Useful environment variables

| Var | Default | Purpose |
|---|---|---|
| `GIFT_EVAL` | (unset) | Path to the GIFT-Eval data directory. |
| `TS_DATASET` | `covid_deaths` | GIFT-Eval dataset name. |
| `TS_EVAL_SERIES` | 0 | Eval series per evaluation (0 = use all). |
| `TS_EVAL_WINDOWS` | 3 | Rolling-origin windows per series. |

## Files

- `evaluator.py` — 2-stage cascade evaluator (validity gate + Chronos-2 MASE).
- `initial_program.py` — seed `TransformProgram` (identity transform).
- `gift_eval_adapter.py` — GIFT-Eval loader, series subsampling, meta computation.
- `config.yaml` — openevolve config (LLM, prompt, MAP-Elites database).

## Notes

- Chronos-2 holds GPU memory; the config sets `parallel_evaluations: 1`.
- The first run downloads the Chronos-2 weights (~500 MB) from the
  Hugging Face Hub.
- Caches (`_pipeline_cache`, `_dataset_cache`, `_pool_baseline_forecasts`)
  are populated on the first evaluation and reused across iterations.
