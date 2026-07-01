"""
TabArena adapter — self-contained loader for TabArena binary classification tasks.

No dependency on the tabular_agent package. Uses the OpenML API and the
publicly hosted TabArena metadata CSV.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

# ── TabArena metadata ────────────────────────────────────────────────────────
_METADATA_URL = (
    "https://raw.githubusercontent.com/autogluon/tabarena/main/"
    "tabarena/tabarena/nips2025_utils/metadata/task_metadata_tabarena51.csv"
)
_CACHE_DIR = Path.home() / ".cache" / "tabarena"


@dataclass
class TaskSplit:
    """One (dataset, fold, repeat) unit of evaluation."""

    tid: int
    dataset: str
    target_col: str
    fold: int
    repeat: int
    n_folds: int
    n_repeats: int
    sample: int = 0


# ── Metadata helpers ─────────────────────────────────────────────────────────

def _get_n_repeats(n_train: int) -> int:
    if n_train < 2_500:
        return 10
    elif n_train < 250_000:
        return 3
    else:
        return 1


def _fetch_metadata_csv() -> pd.DataFrame:
    cache_path = _CACHE_DIR / "task_metadata_tabarena51.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    logger.info("Downloading TabArena metadata from GitHub ...")
    with urlopen(_METADATA_URL) as resp:
        raw = resp.read().decode("utf-8")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(raw)
    return pd.read_csv(io.StringIO(raw))


def _enrich_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["n_folds"] = 3
    df["n_samples_test_per_fold"] = (df["NumberOfInstances"] / 3).astype(int)
    df["n_samples_train_per_fold"] = (
        df["NumberOfInstances"] - df["n_samples_test_per_fold"]
    )
    df["n_repeats"] = df["n_samples_train_per_fold"].apply(_get_n_repeats)
    df["n_features"] = (df["NumberOfFeatures"] - 1).astype(int)
    df["n_classes"] = df["NumberOfClasses"].fillna(0).astype(int)
    df["problem_type"] = df["n_classes"].apply(
        lambda c: "regression" if c == 0 else ("binary" if c == 2 else "multiclass")
    )
    df["dataset"] = df["name"]
    return df


# ── Public API ───────────────────────────────────────────────────────────────

def load_task_metadata(
    max_datasets: Optional[int] = None,
    datasets: Optional[list[str]] = None,
    problem_type: str = "binary",
) -> pd.DataFrame:
    """Load TabArena paper metadata, optionally filtered.

    Parameters
    ----------
    max_datasets : Cap on number of datasets.
    datasets     : Explicit list of dataset names to keep.
    problem_type : Filter to this problem type (default "binary").
    """
    raw = _fetch_metadata_csv()
    meta = _enrich_metadata(raw)
    if problem_type:
        meta = meta[meta["problem_type"] == problem_type]
    if datasets is not None:
        meta = meta[meta["dataset"].isin(datasets)]
    if max_datasets is not None:
        meta = meta.head(max_datasets)
    return meta


def load_openml_split(
    split: TaskSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Fetch a single (fold, repeat) split from the OpenML API.

    Returns
    -------
    (train_df, test_df, target_col, description)
    Both DataFrames include the target column.
    """
    import openml

    logger.info(
        f"Loading OpenML task {split.tid} ({split.dataset}) "
        f"fold={split.fold} repeat={split.repeat}"
    )
    task = openml.tasks.get_task(split.tid, download_splits=True)
    dataset = task.get_dataset()

    description = ""
    try:
        raw_desc = dataset.description or ""
        # TabArena wrapper datasets link to the original OpenML dataset.
        # Try to fetch the richer original description.
        import re
        match = re.search(r"openml\.org/search\?type=data&[^&]*&id=(\d+)", raw_desc)
        if match:
            try:
                orig_ds = openml.datasets.get_dataset(int(match.group(1)))
                description = orig_ds.description or raw_desc
            except Exception:
                description = raw_desc
        else:
            description = raw_desc
    except Exception:
        pass

    X, y, _, _ = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )

    train_idx, test_idx = task.get_train_test_split_indices(
        fold=split.fold,
        repeat=split.repeat,
        sample=split.sample,
    )

    if y.dtype == object or y.dtype.name == "category":
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)

    train_df = X.iloc[train_idx].copy()
    train_df[split.target_col] = y.iloc[train_idx].values

    test_df = X.iloc[test_idx].copy()
    test_df[split.target_col] = y.iloc[test_idx].values

    return train_df, test_df, split.target_col, description


def load_task(
    task_name: str,
    fold: int = 0,
    repeat: int = 0,
) -> dict:
    """Load a single TabArena task by name.

    Returns
    -------
    dict with keys: train_df, test_df, target_col, description, task_info
    """
    meta = load_task_metadata()
    row = meta[meta["dataset"] == task_name]
    if row.empty:
        available = meta["dataset"].tolist()
        raise ValueError(
            f"Task '{task_name}' not found. Available: {available[:20]}"
        )
    row = row.iloc[0]
    split = TaskSplit(
        tid=int(row["tid"]),
        dataset=task_name,
        target_col=str(row["target_feature"]),
        fold=fold,
        repeat=repeat,
        n_folds=int(row["n_folds"]),
        n_repeats=int(row["n_repeats"]),
    )
    train_df, test_df, target_col, description = load_openml_split(split)

    task_info = {
        "task_name": task_name,
        "tid": int(row["tid"]),
        "target_col": target_col,
        "n_classes": int(row["n_classes"]),
        "n_features": int(row["n_features"]),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "problem_type": str(row["problem_type"]),
        "description": description,
    }
    return {
        "train_df": train_df,
        "test_df": test_df,
        "target_col": target_col,
        "description": description,
        "task_info": task_info,
    }


def get_default_task() -> str:
    """Return the default task name for evolution (configurable via env)."""
    return os.environ.get("TABARENA_TASK", "churn")
