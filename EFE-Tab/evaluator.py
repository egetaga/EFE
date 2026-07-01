import os as _os
_os.environ["POSTHOG_DISABLED"] = "true"
_os.environ["TABPFN_NO_ANALYTICS"] = "true"
_os.environ["DO_NOT_TRACK"] = "1"
try:
    import posthog
    posthog.disabled = True
except ImportError:
    pass

import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import importlib.util
import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("analytics").setLevel(logging.CRITICAL)
logging.getLogger("segment").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from openevolve.evaluation_result import EvaluationResult

_DIR = Path(__file__).parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from _evaluator_core import (  # noqa: E402
    FEATURE_EXPLOSION_FACTOR,
    MAX_EXTREME_VALUE,
    SAMPLE_ROWS_STAGE1,
    _build_dataset_context,
    _compute_feature_stats,
    _compute_quality_stats,
    _fit_tabpfn,
    _fit_tabpfn_predict,
    _format_importance_feedback,
    _prepare_for_tabpfn,
)
from helpers import _fit_depth3_lgbm_tree_text  # noqa: E402

from _evaluator_core import _load_task_data  # noqa: E402

HOLDOUT_FRAC = float(os.environ.get("HOLDOUT_FRAC", "0.3"))
_HOLDOUT_BASE_SEED = int(os.environ.get("HOLDOUT_SEED", "42"))
PERM_CV_FOLDS = int(os.environ.get("PERM_CV_FOLDS", "3"))
TREE_HPO_TRIALS = int(os.environ.get("TREE_HPO_TRIALS", "20"))
TREE_CV_FOLDS = int(os.environ.get("TREE_CV_FOLDS", "3"))

ORIGINAL_PERM_CV_FOLDS = int(os.environ.get("ORIGINAL_PERM_CV_FOLDS", "3"))
ORIGINAL_PERM_MAX_ROWS = int(os.environ.get("ORIGINAL_PERM_MAX_ROWS", "5000"))

RELAXED_PENALTY_ALPHA = 0.10

_eval_counter = 0

_stage1_pools_cache: dict | None = None

_original_importance_cache: str | None = None


def _validate_output_relaxed(
    X_in: pd.DataFrame, X_out: pd.DataFrame, cols_before: list[str]
) -> list[str]:
    """Validate transformed output under the RELAXED policy.

    Keeps only structural / safety checks:
      - row count preserved
      - index preserved
      - DataFrame type
      - ≤5× feature-explosion cap
      - all-NaN new columns rejected
      - Inf / extreme numeric values rejected
    """
    issues: list[str] = []

    if len(X_out) != len(X_in):
        issues.append(f"Row count changed: {len(X_in)} -> {len(X_out)}")

    if not X_out.index.equals(X_in.index):
        issues.append("Index not preserved")

    if not isinstance(X_out, pd.DataFrame):
        issues.append(f"Output is {type(X_out)}, expected DataFrame")
        return issues

    if len(X_out.columns) > FEATURE_EXPLOSION_FACTOR * len(X_in.columns):
        issues.append(
            f"Feature explosion: {len(X_in.columns)} -> {len(X_out.columns)} "
            f"(>{FEATURE_EXPLOSION_FACTOR}x)"
        )

    new_cols = [c for c in X_out.columns if c not in cols_before]
    all_nan_new = [c for c in new_cols if X_out[c].isna().all()]
    if all_nan_new:
        issues.append(f"All-NaN generated columns: {all_nan_new[:5]}")

    numeric_out = X_out.select_dtypes(include="number")
    if numeric_out.shape[1] > 0:
        max_abs = numeric_out.abs().max().max()
        if np.isinf(max_abs):
            issues.append("Inf values in output")
        elif max_abs > MAX_EXTREME_VALUE:
            issues.append(f"Extreme values (max abs = {max_abs:.1e})")

    for col in cols_before:
        if col not in X_out.columns or col not in X_in.columns:
            continue
        orig = X_in[col]
        transformed = X_out[col]
        mask = orig.notna() & transformed.notna()
        if mask.any():
            if orig.dtype.kind in ("f", "i"):
                if not np.allclose(
                    orig[mask].values.astype(float),
                    transformed[mask].values.astype(float),
                    equal_nan=True,
                    rtol=1e-7,
                ):
                    issues.append(
                        f"Original column '{col}' values modified in place"
                    )
            else:
                if not (orig[mask].values == transformed[mask].values).all():
                    issues.append(
                        f"Original column '{col}' values modified in place"
                    )
        if not (orig.isna() == transformed.isna()).all():
            issues.append(f"NaN pattern changed in original column '{col}'")

    return issues


def _combined_score(
    delta: float,
    feature_stats: dict,
    n_rows: int,
    transform_time: float,
    valid: bool,
    n_dropped: int = 0,
) -> float:
    """Symmetric **total-change** feature penalty (alpha = 0.10):
        total_changes = n_new + n_dropped
        penalty = alpha * sqrt(total_changes / n_rows)
        score = 0.5 + 5·delta - penalty - time_penalty

    Every drop and every new column costs the same; there is no bonus
    path and no free swap. Do-nothing maps to 0.5 (baseline). Modifying
    an original in place (same column name) is free — it counts as
    neither new nor dropped.
    """
    if not valid:
        return 0.0
    k_new = float(feature_stats["generated_feature_count"])
    total_changes = k_new + float(max(n_dropped, 0))
    if total_changes == 0:
        penalty = 0.0
    else:
        penalty = RELAXED_PENALTY_ALPHA * np.sqrt(total_changes / max(n_rows, 1))
    score = 0.5 + delta * 5.0 - float(penalty)
    if transform_time > 120.0:
        score -= 0.10
    elif transform_time > 60.0:
        score -= 0.05
    return float(np.clip(score, 0.0, 1.5))


def _compute_perm_importance_cv_tabpfn(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    original_cols: list[str],
    candidate_module,
    cv_folds: int,
    seed: int,
) -> dict:
    """Per-fold TabPFN permutation importance on NEW columns, averaged over
    folds, computed on the **full training set**.

    For each of `cv_folds` stratified folds:
      - Fresh feature program → fit on train-fold → transform both folds
        (leak-safe).
      - Fit one TabPFN on the transformed train-fold.
      - Base AUC on the val-fold; for each NEW column permute its values
        on a val-fold copy, predict, record the AUC drop. Average drops
        across folds.

    Only columns whose names are not in `original_cols` are permuted;
    originals that the program kept remain untouched (in-place
    modification is prohibited by the validator).
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    per_col_drops: dict[str, list[float]] = {}
    rng = np.random.RandomState(seed)

    for tr_idx, va_idx in skf.split(X_full, y_full):
        X_tr_fold = X_full.iloc[tr_idx]
        y_tr_fold = y_full.iloc[tr_idx]
        X_va_fold = X_full.iloc[va_idx]
        y_va_fold = y_full.iloc[va_idx]

        try:
            fp_perm = candidate_module.build_feature_program(seed=42)
            state = fp_perm.fit(X_tr_fold, y_tr_fold)
            X_tr_out = fp_perm.transform(X_tr_fold, state)
            X_va_out = fp_perm.transform(X_va_fold, state)
        except Exception as e:
            return {"error": f"fold transform failed: {e}"}

        generated_cols = [c for c in X_va_out.columns if c not in original_cols]
        if not generated_cols:
            return {}

        try:
            clf = _fit_tabpfn(X_tr_out, y_tr_fold, seed=seed)
            X_va_prep = _prepare_for_tabpfn(X_va_out)
            base_proba = clf.predict_proba(X_va_prep)
            base_pos = base_proba[:, 1] if base_proba.shape[1] == 2 else base_proba[:, -1]
            base_score = roc_auc_score(y_va_fold, base_pos)
        except Exception as e:
            return {"error": f"fold TabPFN fit failed: {e}"}

        for col in generated_cols:
            if col not in X_va_prep.columns:
                continue
            X_perm = X_va_prep.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            try:
                perm_proba = clf.predict_proba(X_perm)
                perm_pos = perm_proba[:, 1] if perm_proba.shape[1] == 2 else perm_proba[:, -1]
                perm_score = roc_auc_score(y_va_fold, perm_pos)
                per_col_drops.setdefault(col, []).append(base_score - perm_score)
            except Exception:
                continue

    return {
        col: round(float(np.mean(drops)), 5)
        for col, drops in per_col_drops.items()
    }


def _draw_pools(iter_seed: int) -> dict:
    """Draw a fresh stratified Pool A / Pool B split at the given seed.

    Loads task data via the cached `_load_task_data()` (so OpenML / CSV hits
    disk only once per worker), then performs a stratified (1 − HOLDOUT_FRAC)
    / HOLDOUT_FRAC split.
    """
    task_data = _load_task_data()
    train_df = task_data["train_df"]
    target_col = task_data["target_col"]
    task_info = task_data["task_info"]

    X_full = train_df.drop(columns=[target_col])
    y_full = train_df[target_col]

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=HOLDOUT_FRAC, random_state=iter_seed,
    )
    a_idx, b_idx = next(sss.split(X_full, y_full))

    return {
        "X_A": X_full.iloc[a_idx].reset_index(drop=True),
        "y_A": y_full.iloc[a_idx].reset_index(drop=True),
        "X_B": X_full.iloc[b_idx].reset_index(drop=True),
        "y_B": y_full.iloc[b_idx].reset_index(drop=True),
        "X_full": X_full.reset_index(drop=True),
        "y_full": y_full.reset_index(drop=True),
        "target_col": target_col,
        "task_info": task_info,
    }


def _get_stage1_pools() -> dict:
    """Stage 1 uses a fixed split (HOLDOUT_SEED) so validity is reproducible."""
    global _stage1_pools_cache
    if _stage1_pools_cache is None:
        _stage1_pools_cache = _draw_pools(_HOLDOUT_BASE_SEED)
    return _stage1_pools_cache


def _get_original_column_importance_text() -> str:
    """TabPFN permutation importance for each ORIGINAL column, run once per
    worker on a stratified subsample of **Pool A only** (Pool B is the
    scoring holdout and must not leak into LLM-visible feedback). Returns
    a pre-formatted text block suitable for an artifact. Cached forever.

    Cost control:
      - Stratified subsample up to `ORIGINAL_PERM_MAX_ROWS` rows (default
        5000) before the CV split.
      - `ORIGINAL_PERM_CV_FOLDS` (default 3) — rough signal, one-off.
      - Fixed seed (`_HOLDOUT_BASE_SEED`).
    """
    global _original_importance_cache
    if _original_importance_cache is not None:
        return _original_importance_cache

    pools = _get_stage1_pools()
    X_A, y_A = pools["X_A"], pools["y_A"]

    n_target = min(ORIGINAL_PERM_MAX_ROWS, len(X_A))
    if n_target < len(X_A):
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_target, test_size=None,
            random_state=_HOLDOUT_BASE_SEED,
        )
        idx, _ = next(sss.split(X_A, y_A))
        X = X_A.iloc[idx].reset_index(drop=True)
        y = y_A.iloc[idx].reset_index(drop=True)
    else:
        X = X_A.reset_index(drop=True)
        y = y_A.reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=ORIGINAL_PERM_CV_FOLDS, shuffle=True,
        random_state=_HOLDOUT_BASE_SEED,
    )
    per_col_drops: dict[str, list[float]] = {}
    rng = np.random.RandomState(_HOLDOUT_BASE_SEED)

    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_va = y.iloc[va_idx].reset_index(drop=True)

        try:
            clf = _fit_tabpfn(X_tr, y_tr, seed=_HOLDOUT_BASE_SEED)
            X_va_prep = _prepare_for_tabpfn(X_va)
            base_proba = clf.predict_proba(X_va_prep)
            base_pos = (
                base_proba[:, 1] if base_proba.shape[1] == 2 else base_proba[:, -1]
            )
            base_auc = float(roc_auc_score(y_va, base_pos))
        except Exception as e:
            print(f"  Original-col importance: fold TabPFN failed: {e}")
            continue

        for col in X_va_prep.columns:
            X_perm = X_va_prep.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            try:
                perm_proba = clf.predict_proba(X_perm)
                perm_pos = (
                    perm_proba[:, 1] if perm_proba.shape[1] == 2 else perm_proba[:, -1]
                )
                perm_auc = roc_auc_score(y_va, perm_pos)
                per_col_drops.setdefault(col, []).append(base_auc - perm_auc)
            except Exception:
                continue

    importances = {
        col: round(float(np.mean(drops)), 5)
        for col, drops in per_col_drops.items()
    }

    if not importances:
        _original_importance_cache = "Original-column importance unavailable."
        return _original_importance_cache

    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    lines = [
        f"ORIGINAL column importance — one-shot TabPFN perm-importance on raw "
        f"features (CV {ORIGINAL_PERM_CV_FOLDS}-fold on up to "
        f"{ORIGINAL_PERM_MAX_ROWS} stratified Pool-A rows). AUC drop when "
        f"the column is shuffled; higher = more useful by itself.",
        "",
    ]
    for col, imp in sorted_imp:
        if imp > 0.005:
            label = "USEFUL"
        elif imp > 0.001:
            label = "marginal"
        elif imp > -0.001:
            label = "noise"
        else:
            label = "HARMFUL"
        lines.append(f"  {col}: {imp:+.5f} ({label})")

    _original_importance_cache = "\n".join(lines)
    return _original_importance_cache


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Stage 1: fast validation on a small sample from a fixed Pool A.

    Uses the stage1 (fixed-seed) pool so validity checks are reproducible —
    rotation only matters for stage 2 scoring. Uses the **relaxed** validator
    so candidates that drop originals pass the gate (in-place modification
    still fails).
    """
    artifacts = {}
    try:
        pools = _get_stage1_pools()
        X_A, y_A = pools["X_A"], pools["y_A"]
        task_info = pools["task_info"]

        n_sample = min(SAMPLE_ROWS_STAGE1, len(X_A))
        sample_idx = X_A.sample(n=n_sample, random_state=42).index
        X_sample = X_A.loc[sample_idx]
        y_sample = y_A.loc[sample_idx]

        artifacts["dataset_context"] = _build_dataset_context(X_A, y_A, task_info)

        spec = importlib.util.spec_from_file_location("candidate", program_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)

        if not hasattr(candidate, "build_feature_program"):
            artifacts["error"] = "Missing build_feature_program() function"
            return EvaluationResult(
                metrics={"combined_score": 0.0, "valid": 0.0,
                         "generated_feature_count": 0.0,
                         "n_dropped_original_cols": 0.0},
                artifacts=artifacts,
            )

        fp = candidate.build_feature_program(seed=42)
        for method in ("fit", "transform"):
            if not hasattr(fp, method):
                artifacts["error"] = f"FeatureProgram missing method: {method}"
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "valid": 0.0,
                             "generated_feature_count": 0.0,
                             "n_dropped_original_cols": 0.0},
                    artifacts=artifacts,
                )

        cols_before = X_sample.columns.tolist()
        X_in_copy = X_sample.copy()

        state = fp.fit(X_sample, y_sample)
        if not isinstance(state, dict):
            artifacts["error"] = f"fit() returned {type(state)}, expected dict"
            return EvaluationResult(
                metrics={"combined_score": 0.0, "valid": 0.0,
                         "generated_feature_count": 0.0,
                         "n_dropped_original_cols": 0.0},
                artifacts=artifacts,
            )

        X_out = fp.transform(X_sample, state)

        if not X_in_copy.equals(X_sample):
            artifacts["error"] = "Input DataFrame was mutated in-place"
            return EvaluationResult(
                metrics={"combined_score": 0.0, "valid": 0.0,
                         "generated_feature_count": 0.0,
                         "n_dropped_original_cols": 0.0},
                artifacts=artifacts,
            )

        issues = _validate_output_relaxed(X_sample, X_out, cols_before)
        if issues:
            artifacts["validation_issues"] = "\n".join(issues)
            artifacts["error"] = f"Validation failed: {issues[0]}"
            return EvaluationResult(
                metrics={"combined_score": 0.0, "valid": 0.0,
                         "generated_feature_count": 0.0,
                         "n_dropped_original_cols": 0.0},
                artifacts=artifacts,
            )

        feature_stats = _compute_feature_stats(X_sample, X_out)
        quality_stats = _compute_quality_stats(X_out)

        return EvaluationResult(
            metrics={"combined_score": 0.5, "valid": 1.0,
                     **feature_stats, **quality_stats},
            artifacts=artifacts,
        )

    except Exception as e:
        artifacts["error"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        return EvaluationResult(
            metrics={"combined_score": 0.0, "valid": 0.0,
                     "generated_feature_count": 0.0,
                     "n_dropped_original_cols": 0.0},
            artifacts=artifacts,
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Stage 2: rotating Pool A/B per call, relaxed validation, net-churn
    scoring, CV-based TabPFN perm importance over the full training set.
    LLM-visible feedback still avoids Pool B for the scoring artifacts
    (depth-3 tree is on Pool A); perm importance uses the full training set
    via 3-fold CV (each row appears in the held-out fold for exactly one
    of the CV splits, so the feedback does see Pool B rows — a deliberate
    change from the strict variant per the user's spec).
    """
    global _eval_counter
    _eval_counter += 1
    iter_seed = _HOLDOUT_BASE_SEED + _eval_counter

    artifacts: dict = {}
    try:
        pools = _draw_pools(iter_seed)
        X_A, y_A = pools["X_A"], pools["y_A"]
        X_B, y_B = pools["X_B"], pools["y_B"]
        X_full, y_full = pools["X_full"], pools["y_full"]
        task_info = pools["task_info"]
        cols_before = X_A.columns.tolist()

        spec = importlib.util.spec_from_file_location("candidate", program_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        fp = candidate.build_feature_program(seed=42)

        t0 = time.time()
        state = fp.fit(X_A, y_A)
        X_A_out = fp.transform(X_A, state)
        X_B_out = fp.transform(X_B, state)
        transform_time = time.time() - t0

        issues = _validate_output_relaxed(X_A, X_A_out, cols_before)
        valid = len(issues) == 0

        feature_stats = _compute_feature_stats(X_A, X_A_out)
        quality_stats = _compute_quality_stats(X_A_out)

        try:
            baseline_proba = _fit_tabpfn_predict(X_A, y_A, X_B)
            baseline_auc = float(roc_auc_score(y_B, baseline_proba))
        except Exception as e:
            print(f"  Baseline TabPFN failed: {e}")
            baseline_auc = 0.5

        try:
            proba = _fit_tabpfn_predict(X_A_out, y_A, X_B_out)
            candidate_auc = float(roc_auc_score(y_B, proba))
        except Exception as e:
            print(f"  Candidate TabPFN failed: {e}")
            candidate_auc = 0.5

        delta = candidate_auc - baseline_auc

        new_cols = [c for c in X_A_out.columns if c not in cols_before]
        dropped_cols = [c for c in cols_before if c not in X_A_out.columns]

        score = _combined_score(
            delta, feature_stats, len(X_B),
            transform_time, valid,
            n_dropped=len(dropped_cols),
        )

        try:
            tree_text = _fit_depth3_lgbm_tree_text(
                X_A_out, y_A, cols_before,
                n_trials=TREE_HPO_TRIALS, cv_folds=TREE_CV_FOLDS,
                seed=iter_seed,
            )
        except Exception as e:
            tree_text = f"Decision tree unavailable: {e}"

        try:
            importances = _compute_perm_importance_cv_tabpfn(
                X_A, y_A, cols_before,
                candidate, cv_folds=PERM_CV_FOLDS, seed=iter_seed,
            )
            importance_text = _format_importance_feedback(importances)
        except Exception as e:
            importance_text = f"Permutation importance unavailable: {e}"

        artifacts["dataset_context"] = _build_dataset_context(X_A, y_A, task_info)
        artifacts["stage2_summary"] = json.dumps(
            {
                "iter_id": _eval_counter,
                "iter_seed": iter_seed,
                "model": "tabpfn_holdout_rotating_relaxed",
                "pool_a_rows": len(X_A),
                "pool_b_rows": len(X_B),
                "baseline_auc_on_pool_b": round(baseline_auc, 5),
                "candidate_auc_on_pool_b": round(candidate_auc, 5),
                "delta_vs_baseline": round(delta, 5),
                "combined_score": round(score, 5),
                "new_cols": new_cols,
                "dropped_original_cols": dropped_cols,
                "n_new_cols": len(new_cols),
                "n_dropped_original_cols": len(dropped_cols),
                "n_output_cols": int(X_A_out.shape[1]),
                "valid": valid,
                "validation_issues": issues,
            },
            indent=2,
            default=str,
        )
        artifacts["decision_tree"] = tree_text
        artifacts["permutation_importance"] = importance_text

        try:
            artifacts["original_column_importance"] = (
                _get_original_column_importance_text()
            )
        except Exception as e:
            artifacts["original_column_importance"] = (
                f"Original-column importance unavailable: {e}"
            )

        metrics = {
            "combined_score": score,
            "valid": 1.0 if valid else 0.0,
            "candidate_auc": candidate_auc,
            "baseline_auc": baseline_auc,
            "delta_vs_baseline": delta,
            "transform_time_sec": transform_time,
            "n_new_cols": float(len(new_cols)),
            "n_dropped_original_cols": float(len(dropped_cols)),
            **feature_stats,
            **quality_stats,
        }
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        artifacts["error"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        print(f"  Stage 2 (holdout-rotating RELAXED) EXCEPTION: {e}")
        return EvaluationResult(
            metrics={"combined_score": 0.0, "valid": 0.0,
                     "generated_feature_count": 0.0,
                     "n_dropped_original_cols": 0.0},
            artifacts=artifacts,
        )


def evaluate(program_path: str) -> EvaluationResult:
    r1 = evaluate_stage1(program_path)
    if r1.metrics.get("combined_score", 0) < 0.4:
        return r1
    return evaluate_stage2(program_path)
