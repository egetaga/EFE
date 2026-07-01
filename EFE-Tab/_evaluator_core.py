"""
Shared helpers for the Tabular Feature Evolution evaluator.

The active pipeline lives in ``evaluator.py``; this module supplies the
TabPFN wrappers, task loading, feature/quality stats, dataset-context
builder, and importance-feedback formatter that it imports.
"""

# Disable TabPFN/posthog telemetry (blocks on network timeout)
import os as _os
_os.environ["POSTHOG_DISABLED"] = "true"
_os.environ["TABPFN_NO_ANALYTICS"] = "true"
_os.environ["DO_NOT_TRACK"] = "1"
try:
    import posthog
    posthog.disabled = True
except ImportError:
    pass

# Force 'spawn' so CUDA works in worker processes (must be before torch import)
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
TABPFN_MAX_ROWS = 10_000
FEATURE_EXPLOSION_FACTOR = 5
MAX_EXTREME_VALUE = 1e12
SAMPLE_ROWS_STAGE1 = 200


# ── Task loading ─────────────────────────────────────────────────────────────

_task_data_cache: dict | None = None


def _load_task_data() -> dict:
    """Load the TabArena task configured for this evolution run (cached)."""
    global _task_data_cache
    if _task_data_cache is not None:
        return _task_data_cache
    example_dir = Path(__file__).parent
    sys.path.insert(0, str(example_dir))
    from tabarena_adapter import get_default_task, load_task

    task_name = get_default_task()
    fold = int(os.environ.get("TABARENA_FOLD", "0"))
    repeat = int(os.environ.get("TABARENA_REPEAT", "0"))
    _task_data_cache = load_task(task_name, fold=fold, repeat=repeat)
    return _task_data_cache


# ── TabPFN wrappers ──────────────────────────────────────────────────────────

def _subsample_for_tabpfn(
    X: pd.DataFrame, y: pd.Series, max_rows: int = TABPFN_MAX_ROWS, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample to respect TabPFN row limits."""
    if len(X) <= max_rows:
        return X, y
    rng = np.random.RandomState(seed)
    idx = []
    for label in y.unique():
        label_idx = y[y == label].index.tolist()
        n_keep = max(1, int(max_rows * len(label_idx) / len(y)))
        idx.extend(rng.choice(label_idx, size=min(n_keep, len(label_idx)), replace=False).tolist())
    idx = sorted(idx[:max_rows])
    return X.loc[idx], y.loc[idx]


def _prepare_for_tabpfn(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for TabPFN: cast object/category cols to str,
    leave numeric columns as-is. TabPFN V2 handles mixed types natively
    when given a DataFrame with proper dtypes."""
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == "category":
            out[col] = out[col].astype(str)
        elif out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out


def _fit_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
):
    """Fit TabPFNClassifier and return the fitted model."""
    from tabpfn import TabPFNClassifier

    X_tr, y_tr = _subsample_for_tabpfn(X_train, y_train, seed=seed)
    X_tr_prep = _prepare_for_tabpfn(X_tr)

    clf = TabPFNClassifier(random_state=seed, device="cuda")
    clf.fit(X_tr_prep, y_tr.values)
    return clf


def _predict_proba_class1(clf, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probabilities for class 1."""
    X_prep = _prepare_for_tabpfn(X)
    proba = clf.predict_proba(X_prep)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba[:, -1]


def _fit_tabpfn_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """Fit TabPFNClassifier and return predicted probabilities for class 1."""
    clf = _fit_tabpfn(X_train, y_train, seed=seed)
    return _predict_proba_class1(clf, X_val)


# ── Feature / quality stats ──────────────────────────────────────────────────

def _compute_feature_stats(
    X_in: pd.DataFrame, X_out: pd.DataFrame
) -> dict:
    """Compute feature counts for metrics and MAP-Elites dimensions."""
    input_count = len(X_in.columns)
    new_cols = [c for c in X_out.columns if c not in X_in.columns]

    return {
        "input_feature_count": float(input_count),
        "generated_feature_count": float(len(new_cols)),
        "output_feature_count": float(len(X_out.columns)),
        "generation_ratio": float(len(new_cols) / max(input_count, 1)),
    }


def _compute_quality_stats(X_out: pd.DataFrame) -> dict:
    """NaN/Inf fraction in the output."""
    total_cells = X_out.shape[0] * X_out.shape[1]
    if total_cells == 0:
        return {"nan_fraction_in_output": 0.0, "inf_fraction_in_output": 0.0}

    nan_count = X_out.isna().sum().sum()
    numeric = X_out.select_dtypes(include="number")
    inf_count = np.isinf(numeric.values).sum() if numeric.shape[1] > 0 else 0

    return {
        "nan_fraction_in_output": float(nan_count / total_cells),
        "inf_fraction_in_output": float(inf_count / total_cells),
    }


# ── Dataset context (column names, dtypes, description) ─────────────────────

def _build_dataset_context(X: pd.DataFrame, y: pd.Series, task_info: dict) -> str:
    """Build a concise dataset summary for the LLM: column names, dtypes,
    sample statistics, correlations, class balance, and the dataset description."""
    lines = [
        f"Dataset: {task_info.get('task_name', 'unknown')}",
        f"Rows: {len(X)}, Columns: {len(X.columns)}",
        f"Target: {task_info.get('target_col', '?')} ({task_info.get('problem_type', '?')})",
    ]

    # Class balance
    try:
        counts = y.value_counts()
        total = len(y)
        balance_parts = [f"{v}: {c} ({100*c/total:.0f}%)" for v, c in counts.items()]
        lines.append(f"Class balance: {', '.join(balance_parts)}")
    except Exception:
        pass

    # Prefer domain knowledge markdown file over OpenML description
    task_name = task_info.get("task_name", "")
    domain_knowledge_dir = Path(__file__).parent / "domain_knowledge"
    domain_file = domain_knowledge_dir / f"{task_name}.md"
    if domain_file.exists():
        dk = domain_file.read_text().strip()
        if dk:
            lines.append(f"\nDescription:\n{dk[:5000]}")
        else:
            desc = task_info.get("description", "")
            if desc:
                lines.append(f"\nDescription: {desc[:3000].strip()}")
    else:
        desc = task_info.get("description", "")
        if desc:
            desc_short = desc[:3000].strip()
            if len(desc) > 3000:
                desc_short += "..."
            lines.append(f"\nDescription: {desc_short}")

    lines.append("\nColumns:")
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    for col in X.columns:
        dtype = str(X[col].dtype)
        nunique = X[col].nunique()
        n_missing = X[col].isna().sum()
        missing_pct = f"{100 * n_missing / len(X):.0f}%" if n_missing > 0 else "0%"

        if X[col].dtype.kind in ("f", "i"):
            q25, q50, q75 = X[col].quantile([0.25, 0.5, 0.75])
            skew = X[col].skew()
            lines.append(
                f"  {col} ({dtype}): range [{X[col].min():.3g}, {X[col].max():.3g}], "
                f"Q25={q25:.3g}, median={q50:.3g}, Q75={q75:.3g}, "
                f"skew={skew:.2f}, {nunique} unique, {missing_pct} missing"
            )
        else:
            top_vals = X[col].value_counts().head(3).index.tolist()
            lines.append(f"  {col} ({dtype}): {nunique} unique, "
                         f"top: {top_vals}, {missing_pct} missing")

    # Correlations between numeric columns and target
    if len(numeric_cols) >= 2:
        lines.append("\nCorrelations with target:")
        try:
            target_corr = X[numeric_cols].corrwith(y).sort_values(
                key=abs, ascending=False
            )
            for col, corr in target_corr.items():
                lines.append(f"  {col}: {corr:+.3f}")
        except Exception:
            pass

        # Top inter-feature correlations
        lines.append("\nTop inter-feature correlations:")
        try:
            corr_matrix = X[numeric_cols].corr(method="spearman").abs()
            np.fill_diagonal(corr_matrix.values, 0)
            n_pairs = min(5, len(numeric_cols) * (len(numeric_cols) - 1) // 2)
            for _ in range(n_pairs):
                max_val = corr_matrix.max().max()
                if max_val < 0.1:
                    break
                idx_flat = corr_matrix.values.argmax()
                i, j = divmod(idx_flat, corr_matrix.shape[1])
                c1, c2 = corr_matrix.index[i], corr_matrix.columns[j]
                lines.append(f"  {c1} × {c2}: {max_val:.3f}")
                corr_matrix.iloc[i, j] = 0
                corr_matrix.iloc[j, i] = 0
        except Exception:
            pass

    # Sample rows (first 5, with target)
    lines.append(f"\nSample rows (first 5, with target):")
    try:
        sample_df = X.head(5).copy()
        sample_df[task_info.get("target_col", "target")] = y.head(5).values
        lines.append(sample_df.to_string(max_cols=40))
    except Exception:
        pass

    return "\n".join(lines)


# ── Feedback formatting ──────────────────────────────────────────────────────

def _format_importance_feedback(importances: dict) -> str:
    """Format permutation importance dict as readable text for LLM."""
    if not importances or "error" in importances:
        return f"Permutation importance unavailable: {importances.get('error', 'no features')}"

    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = ["Permutation importance (AUC drop when feature is shuffled):"]
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
    return "\n".join(lines)
