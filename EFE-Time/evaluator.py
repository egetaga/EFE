import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import ast
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

from openevolve.evaluation_result import EvaluationResult

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_CONTEXT_LIMIT = 1024
N_EVAL_SERIES = int(os.environ.get("TS_EVAL_SERIES", "0"))
N_EVAL_WINDOWS = int(os.environ.get("TS_EVAL_WINDOWS", "3"))
N_VALIDITY_SERIES = 10
BATCH_SIZE = 64
ROUND_TRIP_TOL = 1e-4
SCORE_SCALE = 5.0
HARM_PENALTY = float(os.environ.get("TS_HARM_PENALTY", "0"))
WQL_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

COMPLEXITY_ALPHA_LITERALS = float(os.environ.get("TS_ALPHA_LITERALS", "0"))
COMPLEXITY_ALPHA_BRANCHES = float(os.environ.get("TS_ALPHA_BRANCHES", "0"))
COMPLEXITY_ALPHA_PARAMS   = float(os.environ.get("TS_ALPHA_PARAMS",   "0"))
_TRIVIAL_LITERALS = {0, 1, -1, 2, 0.0, 1.0, -1.0, 2.0, 0.5}

_POOL_MULTIPLIER = 5


_pipeline_cache = None
_dataset_cache = None
_eval_series_cache = None
_all_instances_pool = None
_pool_baseline_forecasts = None
_train_samples_cache = None
_dataset_info_cache = None
_eval_counter = 0


def _get_pipeline():
    """Load Chronos-2 pipeline (cached)."""
    global _pipeline_cache
    if _pipeline_cache is None:
        from chronos import BaseChronosPipeline
        logger.info("Loading Chronos-2 model ...")
        _pipeline_cache = BaseChronosPipeline.from_pretrained(
            CHRONOS_MODEL_ID,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        logger.info("Chronos-2 loaded.")
    return _pipeline_cache


def _chronos_forecast_batch(
    history_arrays: list[np.ndarray],
    prediction_length: int,
) -> list[dict]:
    """Run Chronos-2 on a batch of univariate series.

    Returns list of dicts, each with:
        "median": np.ndarray (pred_len,)
        "quantiles": dict mapping quantile level (float) -> np.ndarray (pred_len,)
    """
    pipeline = _get_pipeline()
    all_forecasts = []

    for start in range(0, len(history_arrays), BATCH_SIZE):
        batch = history_arrays[start : start + BATCH_SIZE]
        inputs = []
        for h in batch:
            h_trunc = h[-CHRONOS_CONTEXT_LIMIT:]
            t = torch.tensor(h_trunc, dtype=torch.float32).unsqueeze(0)
            inputs.append({"target": t})

        quantiles_out, _ = pipeline.predict_quantiles(
            inputs,
            prediction_length=prediction_length,
            quantile_levels=WQL_QUANTILE_LEVELS,
        )
        for q_tensor in quantiles_out:
            arr = q_tensor[0].numpy().astype(np.float64)
            q_dict = {}
            for j, qlevel in enumerate(WQL_QUANTILE_LEVELS):
                q_dict[qlevel] = arr[:, j]
            all_forecasts.append({
                "median": q_dict[0.5],
                "quantiles": q_dict,
            })

    return all_forecasts


def _load_data():
    """Load dataset, instance pool, and eval series (cached)."""
    global _dataset_cache, _eval_series_cache, _all_instances_pool
    global _train_samples_cache, _dataset_info_cache

    if _dataset_cache is not None:
        return

    example_dir = Path(__file__).parent
    sys.path.insert(0, str(example_dir))

    from gift_eval_adapter import (
        load_dataset,
        get_seasonality,
        get_eval_series,
        get_training_samples,
        get_default_dataset,
    )

    name = get_default_dataset()
    ds = load_dataset(name)
    season = get_seasonality(ds)

    _dataset_cache = ds
    _train_samples_cache = get_training_samples(ds, n=N_VALIDITY_SERIES)

    if N_EVAL_SERIES > 0:
        pool_size = N_EVAL_SERIES * _POOL_MULTIPLIER
    else:
        pool_size = 0
    _all_instances_pool = get_eval_series(ds, n=pool_size, n_windows=N_EVAL_WINDOWS)

    _eval_series_cache = _all_instances_pool

    _dataset_info_cache = {
        "name": name,
        "prediction_length": ds.prediction_length,
        "seasonal_period": season,
        "freq": ds.freq,
        "n_pool_instances": len(_all_instances_pool),
        "n_eval_per_call": N_EVAL_SERIES if N_EVAL_SERIES > 0 else len(_all_instances_pool),
    }

    eval_per_call = N_EVAL_SERIES if N_EVAL_SERIES > 0 else len(_all_instances_pool)
    logger.info("Dataset: %s, pred_len=%d, season=%d, pool=%d, eval_per_call=%d",
                name, ds.prediction_length, season, len(_all_instances_pool), eval_per_call)


from typing import Optional  # noqa: E402

_dataset_context_cache: Optional[str] = None


def _build_dataset_context(info: dict, train_samples: list) -> str:
    """Build a concise dataset summary for the LLM: metadata, sample series
    statistics, and the domain knowledge markdown (if present).

    Mirrors the pattern from tabular_feature_evolution: prefer a
    domain_knowledge/{name}.md file; otherwise fall back to the AVAILABLE_DATASETS
    description from gift_eval_adapter.
    """
    name = info.get("name", "unknown")
    lines = [
        f"Dataset: {name}",
        f"Frequency: {info.get('freq', '?')}",
        f"Prediction length: {info.get('prediction_length', '?')}",
        f"Seasonal period: {info.get('seasonal_period', '?')}",
        f"Eval series: {info.get('n_eval_series', '?')}",
    ]

    try:
        hist_lens = []
        means = []
        stds = []
        mins = []
        maxs = []
        skews = []
        pos_fracs = []
        zero_fracs = []
        for sample in train_samples:
            y_hist = np.asarray(sample, dtype=float)
            finite = y_hist[np.isfinite(y_hist)]
            if len(finite) == 0:
                continue
            hist_lens.append(len(y_hist))
            means.append(float(np.mean(finite)))
            stds.append(float(np.std(finite)))
            mins.append(float(np.min(finite)))
            maxs.append(float(np.max(finite)))
            pos_fracs.append(float(np.mean(finite > 0)))
            zero_fracs.append(float(np.mean(finite == 0)))
            if len(finite) > 2 and np.std(finite) > 1e-12:
                from scipy import stats as _sps
                skews.append(float(_sps.skew(finite)))
        if hist_lens:
            lines.append("\nSample series statistics (across training samples):")
            lines.append(f"  history length: median={int(np.median(hist_lens))}, "
                         f"range=[{min(hist_lens)}, {max(hist_lens)}]")
            lines.append(f"  mean: median={np.median(means):.3g}, "
                         f"range=[{min(means):.3g}, {max(means):.3g}]")
            lines.append(f"  std:  median={np.median(stds):.3g}, "
                         f"range=[{min(stds):.3g}, {max(stds):.3g}]")
            lines.append(f"  min:  median={np.median(mins):.3g}, "
                         f"range=[{min(mins):.3g}, {max(mins):.3g}]")
            lines.append(f"  max:  median={np.median(maxs):.3g}, "
                         f"range=[{min(maxs):.3g}, {max(maxs):.3g}]")
            lines.append(f"  positive_frac: median={np.median(pos_fracs):.3f}")
            lines.append(f"  zero_frac: median={np.median(zero_fracs):.3f}")
            if skews:
                lines.append(f"  skewness: median={np.median(skews):.3f}, "
                             f"range=[{min(skews):.3f}, {max(skews):.3f}]")
    except Exception as e:
        lines.append(f"\n[stats computation failed: {e}]")

    domain_knowledge_dir = Path(__file__).parent / "domain_knowledge"
    domain_file = domain_knowledge_dir / f"{name}.md"
    if domain_file.exists():
        dk = domain_file.read_text().strip()
        if dk:
            lines.append(f"\nDescription:\n{dk[:5000]}")
            logger.info(
                "[domain_knowledge] loaded %s (%d chars, truncated to 5000)",
                domain_file, len(dk),
            )
        else:
            logger.warning("[domain_knowledge] file exists but is empty: %s", domain_file)
    else:
        logger.warning(
            "[domain_knowledge] NOT FOUND: %s — falling back to AVAILABLE_DATASETS description",
            domain_file,
        )
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from gift_eval_adapter import AVAILABLE_DATASETS
            desc = AVAILABLE_DATASETS.get(name, {}).get("description", "")
            if desc:
                lines.append(f"\nDescription: {desc}")
        except Exception:
            pass

    return "\n".join(lines)


def _get_dataset_context() -> str:
    """Return cached dataset context string, computing it on first call.

    Must be called only after _load_data() has populated the caches.
    """
    global _dataset_context_cache
    if _dataset_context_cache is None:
        info = _dataset_info_cache
        samples = _train_samples_cache
        assert info is not None, "_load_data() must run first"
        assert samples is not None, "_load_data() must run first"
        _dataset_context_cache = _build_dataset_context(info, samples)
    return _dataset_context_cache


def _seasonal_naive_error(y_hist: np.ndarray, seasonal_period: int) -> float:
    """Mean absolute error of a seasonal naive forecast on history."""
    if len(y_hist) <= seasonal_period:
        if len(y_hist) <= 1:
            return 0.0
        return float(np.nanmean(np.abs(np.diff(y_hist))))
    diffs = np.abs(y_hist[seasonal_period:] - y_hist[:-seasonal_period])
    return float(np.nanmean(diffs))


def _mase_single(
    actual: np.ndarray,
    forecast: np.ndarray,
    y_hist: np.ndarray,
    seasonal_period: int,
) -> float:
    """MASE for a single series. Returns NaN if scale is ~0."""
    scale = _seasonal_naive_error(y_hist, seasonal_period)
    if scale < 1e-10:
        return np.nan
    return float(np.nanmean(np.abs(actual - forecast)) / scale)


def _quantile_loss_single(
    actual: np.ndarray,
    q_forecast: np.ndarray,
    q_level: float,
) -> float:
    """Quantile loss for one quantile level on one series.

    QL_q = 2 * |( actual - forecast ) * ( indicator(forecast >= actual) - q )|
    """
    error = actual - q_forecast
    loss = 2.0 * np.abs(error * (np.where(q_forecast >= actual, 1.0, 0.0) - q_level))
    return float(np.nansum(loss))


def _aggregate_metrics(
    actuals: list[np.ndarray],
    forecast_dicts: list[dict],
    histories: list[np.ndarray],
    seasonal_period: int,
) -> dict:
    """Compute MASE, wQL, and MAE across all series.

    Parameters
    ----------
    actuals : list of 1-D arrays (labels)
    forecast_dicts : list of dicts from _chronos_forecast_batch, each with
        "median" and "quantiles" keys
    histories : list of 1-D arrays (input history for each series)
    seasonal_period : int

    Returns
    -------
    dict with mase, wql, mae, per_series_mase, n_valid, etc.
    """
    per_series_mase = []
    per_series_mae = []
    total_abs_target = 0.0
    per_level_ql_sum = {q: 0.0 for q in WQL_QUANTILE_LEVELS}

    for actual, fc_dict, hist in zip(actuals, forecast_dicts, histories):
        median = fc_dict["median"]

        m = _mase_single(actual, median, hist, seasonal_period)
        per_series_mase.append(m)

        per_series_mae.append(float(np.nanmean(np.abs(actual - median))))

        abs_target = float(np.nansum(np.abs(actual)))
        total_abs_target += abs_target
        for q_level, q_forecast in fc_dict["quantiles"].items():
            per_level_ql_sum[q_level] += _quantile_loss_single(actual, q_forecast, q_level)

    per_series_mase = np.array(per_series_mase)
    per_series_mae = np.array(per_series_mae)
    valid_mase = per_series_mase[np.isfinite(per_series_mase)]

    if total_abs_target > 1e-10:
        wql = float(np.mean([
            per_level_ql_sum[q] / total_abs_target for q in WQL_QUANTILE_LEVELS
        ]))
    else:
        wql = np.nan

    result = {
        "n_valid": int(len(valid_mase)),
        "per_series_mase": per_series_mase,
    }

    if len(valid_mase) > 0:
        result["mase"] = float(np.mean(valid_mase))
        result["median_mase"] = float(np.median(valid_mase))
        result["std_mase"] = float(np.std(valid_mase))
    else:
        result["mase"] = np.nan

    result["mae"] = float(np.nanmean(per_series_mae))
    result["wql"] = wql

    return result


def _precompute_pool_baseline():
    """Compute Chronos baseline forecasts for the entire instance pool (once)."""
    global _pool_baseline_forecasts
    if _pool_baseline_forecasts is not None:
        return

    _load_data()
    pred_len = _dataset_info_cache["prediction_length"]

    logger.info("Precomputing baseline for %d pool instances ...", len(_all_instances_pool))
    histories = [s["input_target"] for s in _all_instances_pool]

    t0 = time.time()
    _pool_baseline_forecasts = _chronos_forecast_batch(histories, pred_len)
    elapsed = time.time() - t0

    logger.info("Pool baseline precomputed in %.1fs", elapsed)


def _baseline_for_indices(indices: list[int]) -> dict:
    """Aggregate baseline metrics for a subset of the pool."""
    season = _dataset_info_cache["seasonal_period"]
    actuals = [_all_instances_pool[i]["label_target"] for i in indices]
    forecasts = [_pool_baseline_forecasts[i] for i in indices]
    histories = [_all_instances_pool[i]["input_target"] for i in indices]
    return _aggregate_metrics(actuals, forecasts, histories, season)


def _sample_eval_subset() -> tuple[list[dict], list[int]]:
    """Return a rotating subsample of eval instances + their indices.

    When N_EVAL_SERIES > 0 and pool is larger, each call draws a different
    random subset (seed increments). Otherwise returns the full pool.
    """
    global _eval_counter
    _eval_counter += 1

    pool_size = len(_all_instances_pool)
    if N_EVAL_SERIES > 0 and pool_size > N_EVAL_SERIES:
        rng = np.random.RandomState(42 + _eval_counter)
        indices = rng.choice(pool_size, size=N_EVAL_SERIES, replace=False)
        indices.sort()
        indices = indices.tolist()
    else:
        indices = list(range(pool_size))

    eval_series = [_all_instances_pool[i] for i in indices]
    return eval_series, indices


def _validate_transform(
    program,
    train_samples: list[np.ndarray],
    info: dict,
    eval_series: list[dict] | None = None,
) -> list[str]:
    """Run validity checks on the transform. Returns list of issues.

    Uses train_samples for basic checks, plus a subsample of eval_series
    for the forecast-length inverse check (eval series have more diverse
    characteristics and better represent what the transform will face).
    """
    from gift_eval_adapter import compute_meta

    issues = []
    pred_len = info["prediction_length"]
    season = info["seasonal_period"]

    all_samples = list(train_samples)
    if eval_series:
        rng = np.random.RandomState(99)
        n_extra = min(20, len(eval_series))
        idx = rng.choice(len(eval_series), size=n_extra, replace=False)
        for j in idx:
            all_samples.append(eval_series[j]["input_target"])

    for i, y_hist in enumerate(all_samples):
        try:
            meta = compute_meta(y_hist, pred_len, season)
            state = program.fit(y_hist, meta=meta)
        except Exception as e:
            issues.append(f"fit() failed on series {i}: {e}")
            continue

        if not isinstance(state, dict):
            issues.append(f"fit() returned {type(state)}, expected dict")
            continue

        try:
            z = program.transform(y_hist, state)
        except Exception as e:
            issues.append(f"transform() failed on series {i}: {e}")
            continue

        if len(z) != len(y_hist):
            issues.append(f"Series {i}: transform changed length {len(y_hist)} -> {len(z)}")
            continue

        input_nan = np.isnan(y_hist)
        output_nan = np.isnan(z)
        output_inf = np.isinf(z)
        if output_inf.any():
            issues.append(f"Series {i}: transform produced {output_inf.sum()} Inf values")
            continue

        new_nans = output_nan & ~input_nan
        if new_nans.any():
            issues.append(f"Series {i}: transform introduced {new_nans.sum()} new NaN values")
            continue

        z_valid = z[np.isfinite(z)]
        y_valid = y_hist[np.isfinite(y_hist)]
        if len(z_valid) > 1 and np.std(z_valid) < 1e-10 and np.std(y_valid) > 1e-10:
            issues.append(f"Series {i}: transform collapsed to constant")
            continue


        if len(z_valid) > 1:
            z_min, z_max = np.min(z_valid), np.max(z_valid)
            z_range = z_max - z_min
            if z_range > 1e-10:
                z_extended = np.array([
                    z_min - 0.1 * z_range,
                    z_max + 0.1 * z_range,
                ])
                try:
                    y_ext = program.inverse_transform(z_extended, state)
                    if not np.all(np.isfinite(y_ext)):
                        issues.append(
                            f"Series {i}: inverse unstable beyond observed range"
                        )
                except Exception as e:
                    issues.append(
                        f"Series {i}: inverse_transform fails beyond range: {e}"
                    )

        if len(y_hist) > 2 * pred_len:
            y_fit_short = y_hist[:-pred_len]
            try:
                meta_short = compute_meta(y_fit_short, pred_len, season)
                state_short = program.fit(y_fit_short, meta=meta_short)
                if not isinstance(state_short, dict):
                    issues.append(
                        f"Series {i}: fit() returned non-dict on shortened history"
                    )
                    continue
                z_full_short = program.transform(y_hist, state_short)
                if len(z_full_short) != len(y_hist):
                    issues.append(
                        f"Series {i}: transform length mismatch on shortened-fit "
                        f"({len(y_hist)} -> {len(z_full_short)})"
                    )
                    continue
                z_forecast_proxy = np.asarray(z_full_short[-pred_len:]).copy()
                y_forecast_proxy = program.inverse_transform(z_forecast_proxy, state_short)
            except Exception as e:
                issues.append(
                    f"Series {i}: forecast-length inverse_transform failed: {e}"
                )
                continue

            if len(y_forecast_proxy) != pred_len:
                issues.append(
                    f"Series {i}: forecast-length inverse changed length "
                    f"{pred_len} -> {len(y_forecast_proxy)}"
                )
                continue
            if not np.all(np.isfinite(y_forecast_proxy[np.isfinite(z_forecast_proxy)])):
                issues.append(
                    f"Series {i}: forecast-length inverse produced non-finite values"
                )
                continue

            y_tail = y_hist[-pred_len:]
            tail_mask = np.isfinite(y_tail) & np.isfinite(y_forecast_proxy)
            if tail_mask.any():
                tail_err = np.max(np.abs(
                    y_tail[tail_mask] - y_forecast_proxy[tail_mask]
                ))
                tail_scale = max(np.max(np.abs(y_tail[tail_mask])), 1.0)
                if tail_err / tail_scale > ROUND_TRIP_TOL:
                    issues.append(
                        f"Series {i}: forecast-length inverse round-trip "
                        f"error {tail_err:.2e} (scale={tail_scale:.2e}) — "
                        f"inverse_transform does not correctly use state "
                        f"['hist_len'] + np.arange(len(z)) for forecast positions"
                    )

    return issues


def _evaluate_forecast(program, eval_series: list[dict], info: dict) -> dict:
    """Run Chronos-2 with the candidate transform, compute MASE, wQL, MAE."""
    from gift_eval_adapter import compute_meta

    pred_len = info["prediction_length"]
    season = info["seasonal_period"]

    transformed_histories = []
    transform_states = []
    transform_errors = []
    param_counts = []
    t0_transform = time.time()

    for s in eval_series:
        y_hist = s["input_target"]
        try:
            meta = compute_meta(y_hist, pred_len, season)
            state = program.fit(y_hist, meta=meta)
            z_hist = program.transform(y_hist, state)
            z_hist = np.where(np.isinf(z_hist), np.nan, z_hist)
            transformed_histories.append(z_hist)
            transform_states.append(state)
            transform_errors.append(None)
            param_counts.append(len([k for k in state if k != "hist_len"]))
        except Exception as e:
            transformed_histories.append(y_hist.copy())
            transform_states.append({"family": "identity_fallback"})
            transform_errors.append(str(e))

    transform_time = time.time() - t0_transform

    t0_forecast = time.time()
    z_forecast_dicts = _chronos_forecast_batch(transformed_histories, pred_len)
    forecast_time = time.time() - t0_forecast

    forecast_dicts = []
    inverse_errors = []
    for i, (z_fc_dict, state) in enumerate(zip(z_forecast_dicts, transform_states)):
        if transform_errors[i] is not None:
            forecast_dicts.append(z_fc_dict)
            inverse_errors.append(None)
            continue
        try:
            inv_median = program.inverse_transform(z_fc_dict["median"], state)
            inv_median = np.where(np.isinf(inv_median), np.nan, inv_median)
            inv_quantiles = {}
            for q_level, z_q in z_fc_dict["quantiles"].items():
                inv_q = program.inverse_transform(z_q, state)
                inv_quantiles[q_level] = np.where(np.isinf(inv_q), np.nan, inv_q)
            forecast_dicts.append({"median": inv_median, "quantiles": inv_quantiles})
            inverse_errors.append(None)
        except Exception as e:
            forecast_dicts.append(z_fc_dict)
            inverse_errors.append(str(e))

    actuals = [s["label_target"] for s in eval_series]
    histories = [s["input_target"] for s in eval_series]
    metrics_result = _aggregate_metrics(actuals, forecast_dicts, histories, season)

    n_transform_errors = sum(1 for e in transform_errors if e is not None)
    n_inverse_errors = sum(1 for e in inverse_errors if e is not None)

    return {
        **metrics_result,
        "transform_time_sec": transform_time,
        "forecast_time_sec": forecast_time,
        "n_transform_errors": n_transform_errors,
        "n_inverse_errors": n_inverse_errors,
        "max_n_params": max(param_counts) if param_counts else 0,
    }


def _count_code_complexity(program_path: str) -> dict:
    """Count complexity indicators in the program source via AST.

    Returns dict with n_literals (non-trivial numeric constants) and
    n_branches (if/elif nodes).
    """
    try:
        source = Path(program_path).read_text()
        tree = ast.parse(source)
    except Exception:
        return {"n_literals": 0, "n_branches": 0}

    n_literals = 0
    n_branches = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            n_branches += 1

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in _TRIVIAL_LITERALS:
                n_literals += 1

    return {"n_literals": n_literals, "n_branches": n_branches}


def _compute_harm_rate(
    per_series_cand: np.ndarray,
    per_series_base: np.ndarray,
) -> float:
    """Fraction of series where the candidate is worse than baseline."""
    valid = np.isfinite(per_series_cand) & np.isfinite(per_series_base)
    if not valid.any():
        return 0.0
    ratios = per_series_cand[valid] / np.maximum(per_series_base[valid], 1e-10)
    return float(np.mean(ratios > 1.0))


def _combined_score(
    candidate_mase: float,
    baseline_mase: float,
    harm_rate: float = 0.0,
    complexity_penalty: float = 0.0,
) -> float:
    """Score = 0.5 + (1 - candidate/baseline) * SCALE - harm² * coeff - complexity.

    Identity -> 0.5; 10% MASE reduction -> 1.0; 10% increase -> 0.0.
    Squared harm rate is lenient when few series are harmed (natural) but
    punishes heavily when harm is widespread (overfitting signal).
    Complexity penalty further discourages overfit code with many magic
    constants, branches, or fitted parameters.
    """
    if np.isnan(candidate_mase) or np.isnan(baseline_mase) or baseline_mase < 1e-10:
        return 0.0
    ratio = candidate_mase / baseline_mase
    score = 0.5 + (1.0 - ratio) * SCORE_SCALE
    score -= harm_rate ** 2 * HARM_PENALTY
    score -= complexity_penalty
    return float(np.clip(score, 0.0, 1.5))


def _build_artifacts(
    info: dict,
    stage1_issues: list[str] | None,
    candidate_result: dict | None,
    baseline_result: dict | None,
    score: float,
    complexity_info: dict | None = None,
) -> str:
    """Build a JSON artifact string for the LLM."""
    artifact = {
        "dataset": info["name"],
        "prediction_length": info["prediction_length"],
        "seasonal_period": info["seasonal_period"],
        "score": score,
    }

    if stage1_issues:
        artifact["stage1_issues"] = stage1_issues[:10]

    if baseline_result is not None:
        artifact["baseline_mase"] = baseline_result["mase"]
        artifact["baseline_wql"] = baseline_result.get("wql")
        artifact["baseline_mae"] = baseline_result.get("mae")

    if candidate_result is not None:
        cand_mase = candidate_result["mase"]
        base_mase = baseline_result["mase"] if baseline_result else np.nan

        artifact["candidate_mase"] = cand_mase
        artifact["candidate_wql"] = candidate_result.get("wql")
        artifact["candidate_mae"] = candidate_result.get("mae")
        artifact["mase_ratio"] = (
            cand_mase / base_mase if base_mase > 1e-10 else None
        )
        base_wql = baseline_result.get("wql") if baseline_result else None
        cand_wql = candidate_result.get("wql")
        if base_wql and cand_wql and base_wql > 1e-10:
            artifact["wql_ratio"] = cand_wql / base_wql
        artifact["n_valid_series"] = candidate_result["n_valid"]
        artifact["n_transform_errors"] = candidate_result.get("n_transform_errors", 0)
        artifact["n_inverse_errors"] = candidate_result.get("n_inverse_errors", 0)
        artifact["transform_time_sec"] = round(candidate_result.get("transform_time_sec", 0), 2)
        artifact["forecast_time_sec"] = round(candidate_result.get("forecast_time_sec", 0), 2)

        per_series = candidate_result.get("per_series_mase", np.array([]))
        base_per_series = baseline_result.get("per_series_mase", np.array([])) if baseline_result else np.array([])

        if len(per_series) > 0 and len(base_per_series) == len(per_series):
            valid_mask = np.isfinite(per_series) & np.isfinite(base_per_series)
            if valid_mask.any():
                ratios = per_series[valid_mask] / np.maximum(base_per_series[valid_mask], 1e-10)
                artifact["harm_rate"] = float(np.mean(ratios > 1.0))
                artifact["help_rate"] = float(np.mean(ratios < 1.0))
                artifact["median_mase_ratio"] = float(np.median(ratios))
                best_idx = np.argmin(ratios)
                worst_idx = np.argmax(ratios)
                artifact["best_series_ratio"] = float(ratios[best_idx])
                artifact["worst_series_ratio"] = float(ratios[worst_idx])

    if complexity_info is not None:
        artifact["complexity"] = complexity_info

    return json.dumps(artifact, indent=2, default=str)


def _load_program(program_path: str):
    """Load and return a TransformProgram from a file path."""
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_transform_program(seed=42)


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Stage 1: Fast validity gate (no forecasting).

    Checks: load, fit, transform, inverse_transform, round-trip, stability.
    Returns combined_score=0.5 if valid (passes to Stage 2), 0.0 if invalid.
    """
    _load_data()
    info = _dataset_info_cache
    dataset_context = _get_dataset_context()

    try:
        program = _load_program(program_path)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Stage 1 FAIL: Cannot load program: %s", e)
        artifact = json.dumps({
            "error": "load_failed",
            "message": str(e),
            "traceback": tb[-500:],
        })
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "evaluation_summary": artifact,
                "dataset_context": dataset_context,
            },
        )

    logger.info("Stage 1: Validity checks ...")
    t0 = time.time()
    issues = _validate_transform(program, _train_samples_cache, info, _eval_series_cache)
    stage1_time = time.time() - t0
    logger.info("  Time: %.2fs, Issues: %d", stage1_time, len(issues))

    if issues:
        logger.info("  FAIL: %s", issues[0])
        artifact_str = _build_artifacts(info, issues, None, None, 0.0)
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "evaluation_summary": artifact_str,
                "dataset_context": dataset_context,
            },
        )

    logger.info("  PASS")
    return EvaluationResult(
        metrics={"combined_score": 0.5},
        artifacts={
            "evaluation_summary": json.dumps({"stage1": "passed"}),
            "dataset_context": dataset_context,
        },
    )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Stage 2: Chronos-2 forecast MASE evaluation.

    Fits transform per series, forecasts in transformed space,
    inverse-transforms, computes MASE relative to identity baseline.
    """
    _load_data()
    info = _dataset_info_cache
    dataset_context = _get_dataset_context()

    try:
        program = _load_program(program_path)
    except Exception as e:
        logger.error("Stage 2 FAIL: Cannot load program: %s", e)
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "evaluation_summary": json.dumps({"error": str(e)}),
                "dataset_context": dataset_context,
            },
        )

    logger.info("Stage 2: Forecast evaluation ...")

    _precompute_pool_baseline()
    eval_series, eval_indices = _sample_eval_subset()
    baseline = _baseline_for_indices(eval_indices)

    t0 = time.time()
    candidate = _evaluate_forecast(program, eval_series, info)
    stage2_time = time.time() - t0

    cand_mase = candidate["mase"]
    base_mase = baseline["mase"]

    harm_rate = _compute_harm_rate(
        candidate.get("per_series_mase", np.array([])),
        baseline.get("per_series_mase", np.array([])),
    )

    code_complexity = _count_code_complexity(program_path)
    max_n_params = candidate.get("max_n_params", 0)
    n_pool = len(_all_instances_pool)
    complexity_scale = 1.0 / np.sqrt(max(n_pool, 1))
    complexity_penalty = complexity_scale * (
        COMPLEXITY_ALPHA_LITERALS * code_complexity["n_literals"]
        + COMPLEXITY_ALPHA_BRANCHES * code_complexity["n_branches"]
        + COMPLEXITY_ALPHA_PARAMS * max_n_params
    )

    score = _combined_score(cand_mase, base_mase, harm_rate, complexity_penalty)

    logger.info("  Candidate MASE: %.4f, wQL: %.4f, MAE: %.4f",
                cand_mase, candidate['wql'], candidate['mae'])
    logger.info("  Baseline MASE: %.4f, wQL: %.4f, MAE: %.4f",
                base_mase, baseline['wql'], baseline['mae'])
    if base_mase > 1e-10:
        logger.info("  MASE ratio: %.4f, harm_rate: %.2f%%",
                    cand_mase / base_mase, harm_rate * 100)
    else:
        logger.info("  Ratio: N/A")
    logger.info("  Complexity: %d literals, %d branches, %d params → penalty=%.4f",
                code_complexity['n_literals'], code_complexity['n_branches'],
                max_n_params, complexity_penalty)
    logger.info("  Score: %.4f", score)
    logger.info("  Time: %.1fs (transform=%.1fs, forecast=%.1fs)",
                stage2_time, candidate['transform_time_sec'],
                candidate['forecast_time_sec'])

    artifact_str = _build_artifacts(
        info, None, candidate, baseline, score,
        complexity_info={
            "n_literals": code_complexity["n_literals"],
            "n_branches": code_complexity["n_branches"],
            "n_fitted_params": max_n_params,
            "penalty": round(complexity_penalty, 4),
        },
    )

    return EvaluationResult(
        metrics={
            "combined_score": score,
            "mase": cand_mase,
            "wql": candidate["wql"],
            "mae": candidate["mae"],
        },
        artifacts={
            "evaluation_summary": artifact_str,
            "dataset_context": dataset_context,
        },
    )


def evaluate(program_path: str) -> EvaluationResult:
    """Run both stages sequentially. Used for standalone testing."""
    result1 = evaluate_stage1(program_path)
    if result1.metrics.get("combined_score", 0.0) < 0.3:
        return result1
    return evaluate_stage2(program_path)
