"""
GIFT-Eval adapter — dataset loading, series subsampling, and meta computation
for time-series transform evolution.

Set the GIFT_EVAL env var to the directory containing downloaded GIFT-Eval data.
Set TS_DATASET env var to choose which dataset (default: hospital).
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Suppress pandas FutureWarnings from gift_eval / gluonts freq handling
warnings.filterwarnings("ignore", category=FutureWarning, module="gift_eval")
warnings.filterwarnings("ignore", category=FutureWarning, module="gluonts")

# ── Dataset config ──────────────────────────────────────────────────────────

AVAILABLE_DATASETS = {
    # key = TS_DATASET env var value (also used for domain_knowledge/{key}.md)
    # "gift_eval_name" = name passed to Dataset() if different from key (for nested dirs)
    # Yearly
    "m4_yearly": {"term": "short", "description": "23000 yearly M4 series"},
    # Monthly or Quarterly
    "m4_quarterly": {"term": "short", "description": "24000 quarterly M4 series"},
    "hospital": {"term": "short", "description": "767 monthly hospital series"},
    "m4_monthly": {"term": "short", "description": "48000 monthly M4 series"},
    "car_parts_with_missing": {"term": "short", "description": "2674 monthly intermittent demand, zeros + spikes"},
    # Weekly
    "electricity/W": {"term": "short", "description": "370 weekly electricity demand"},
    "solar/W": {"term": "short", "description": "137 weekly solar power"},
    "m4_weekly": {"term": "short", "description": "359 weekly M4 series"},
    "hierarchical_sales_W": {"term": "short", "gift_eval_name": "hierarchical_sales/W", "description": "Weekly hierarchical retail sales (W-WED)"},
    # Daily
    "m4_daily": {"term": "short", "description": "4227 daily M4 series"},
    "jena_weather_D": {"term": "short", "gift_eval_name": "jena_weather/D", "univariate": True, "description": "21 variables, daily weather data from Jena"},
    "covid_deaths": {"term": "short", "description": "266 daily COVID death counts, extreme spikes + regime changes"},
    "electricity_D": {"term": "short", "gift_eval_name": "electricity/D", "description": "370 daily electricity demand series"},
    "solar_D": {"term": "short", "gift_eval_name": "solar/D", "description": "137 daily solar power, bounded + weather-driven volatility"},
    "us_births_D": {"term": "short", "gift_eval_name": "us_births/D", "description": "US daily birth counts (smooth, weekly + annual seasonality)"},
    "restaurant": {"term": "short", "description": "Restaurant daily reservations / sales"},
    "temperature_rain_with_missing": {"term": "short", "description": "Daily temperature and rainfall (zero-inflated, missingness)"},
    # Hourly
    "electricity_H": {"term": "short", "gift_eval_name":"electricity/H", "description": "370 hourly electricity, strong daily/weekly cycles + demand spikes"},
    "solar_H": {"term": "short", "gift_eval_name":"solar/H", "description": "137 hourly solar power, sunrise/sunset ramps + weather spikes"},
    "m4_hourly": {"term": "short", "description": "414 hourly M4 series"},
    "jena_weather_H": {"term": "short", "gift_eval_name": "jena_weather/H", "univariate": True, "description": "21 variables, hourly weather data from Jena"},
    "bitbrains_fast_storage_H": {"term": "short", "gift_eval_name": "bitbrains_fast_storage/H", "univariate": True, "description": "1250 hourly cloud-storage workload series (multivariate read+write, exploded to 2500 univariate)"},
    "bitbrains_rnd_H": {"term": "short", "gift_eval_name": "bitbrains_rnd/H", "univariate": True, "description": "Hourly cloud-VM workload (multivariate read+write, exploded to univariate); sister of bitbrains_fast_storage"},
    "ett1_H": {"term": "short", "gift_eval_name": "ett1/H", "univariate": True, "description": "ETTh1: 7-variate hourly oil-temp transformer (multivariate, exploded to univariate)"},
    "ett2_H": {"term": "short", "gift_eval_name": "ett2/H", "univariate": True, "description": "ETTh2: 7-variate hourly oil-temp transformer (multivariate, exploded to univariate)"},
    "kdd_cup_2018_H": {"term": "short", "gift_eval_name": "kdd_cup_2018_with_missing/H", "description": "KDD Cup 2018 hourly air quality (PM2.5, missingness)"},
    "bizitobs_l2c_H": {"term": "short", "gift_eval_name": "bizitobs_l2c/H", "univariate": True, "description": "BizITObs Layer-2-Cloud: 7-variate hourly cloud KPIs (multivariate, exploded to univariate)"},
    "M_DENSE_H": {"term": "short", "gift_eval_name": "M_DENSE/H", "description": "Melbourne hourly traffic (M-DENSE)"},
    # Minute
    "jena_weather_10T": {"term": "short", "gift_eval_name": "jena_weather/10T", "univariate": True, "description": "21 variables, 10-min frequency weather data from Jena"},
    # Traffic
    "LOOP_SEATTLE_H": {"term": "short", "gift_eval_name": "LOOP_SEATTLE/H", "description": "323 hourly traffic loop detectors, rush-hour spikes"},
    "LOOP_SEATTLE_D": {"term": "short", "gift_eval_name": "LOOP_SEATTLE/D", "description": "323 daily traffic loop detectors"},
    "LOOP_SEATTLE_5T": {"term": "short", "gift_eval_name": "LOOP_SEATTLE/5T", "description": "323 5-min traffic loop detectors, very bursty"},
    "SZ_TAXI_H": {"term": "short", "gift_eval_name": "SZ_TAXI/H", "description": "156 hourly taxi demand, event-driven spikes"},
    "SZ_TAXI_15T": {"term": "short", "gift_eval_name": "SZ_TAXI/15T", "description": "156 15-min taxi demand, very bursty"},
}


def get_default_dataset() -> str:
    return os.environ.get("TS_DATASET", "hospital")


def load_dataset(name: str | None = None):
    """Load a GIFT-Eval Dataset object."""
    from gift_eval.data import Dataset

    if name is None:
        name = get_default_dataset()
    info = AVAILABLE_DATASETS.get(name)
    term = info["term"] if info else "short"
    # Use gift_eval_name for nested directory structures (e.g. jena_weather/D)
    dataset_name = info.get("gift_eval_name", name) if info else name
    # Multivariate datasets need to_univariate=True so each variate becomes a
    # separate 1-D series (the evaluator and TransformProgram expect 1-D input)
    to_univariate = info.get("univariate", False) if info else False
    logger.info(f"Loading GIFT-Eval dataset: {name} (gift_eval_name={dataset_name}, term={term}, to_univariate={to_univariate})")
    return Dataset(name=dataset_name, term=term, to_univariate=to_univariate)


def get_seasonality(dataset) -> int:
    """Return seasonal period for a dataset."""
    from gluonts.time_feature import get_seasonality as gluon_seasonality
    return gluon_seasonality(dataset.freq)


# ── Series subsampling ──────────────────────────────────────────────────────

def get_eval_series(
    dataset, n: int = 200, n_windows: int = 3, seed: int = 42,
) -> list[dict]:
    """Return evaluation instances with multiple rolling forecast origins.

    Creates *n_windows* non-overlapping forecast origins per series from the
    validation split, then subsamples to *n* total instances. Using multiple
    origins prevents overfitting to a single time period.

    Each element is a dict with:
        input_target: np.ndarray (1-D history)
        label_target: np.ndarray (1-D future)
        item_id: str
    """
    pred_len = dataset.prediction_length
    all_instances = []
    for i, entry in enumerate(dataset.validation_dataset):
        target = np.asarray(entry["target"], dtype=np.float64)
        # Need at least n_windows * pred_len + some minimum history
        min_history = max(pred_len, 12)  # at least one seasonal period
        min_len = min_history + n_windows * pred_len
        if len(target) < min_history + pred_len:
            continue  # too short even for 1 window

        # Create up to n_windows origins, stepping back by pred_len each time
        actual_windows = min(n_windows, (len(target) - min_history) // pred_len)
        for w in range(actual_windows):
            # Window w=0 is the most recent, w=1 is one step back, etc.
            end = len(target) - w * pred_len
            label_start = end - pred_len
            if label_start < min_history:
                break
            all_instances.append({
                "input_target": target[:label_start],
                "label_target": target[label_start:end],
                "item_id": f"{entry.get('item_id', i)}_w{w}",
            })

    if n <= 0 or len(all_instances) <= n:
        return all_instances

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_instances), size=n, replace=False)
    idx.sort()
    return [all_instances[i] for i in idx]


def get_test_series(dataset, n: int = 200, seed: int = 42) -> list[dict]:
    """Return a fixed subsample of TEST instances for final evaluation only.

    Uses the official gift-eval test split with rolling windows.
    Should NOT be used during evolution — only for evaluating the final winner.

    Each element is a dict with:
        input_target: np.ndarray (1-D history)
        label_target: np.ndarray (1-D future)
        item_id: str
    """
    all_instances = []
    for inp, label in dataset.test_data:
        all_instances.append({
            "input_target": np.asarray(inp["target"], dtype=np.float64),
            "label_target": np.asarray(label["target"], dtype=np.float64),
            "item_id": str(inp.get("item_id", len(all_instances))),
        })

    if n <= 0 or len(all_instances) <= n:
        return all_instances

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_instances), size=n, replace=False)
    idx.sort()
    return [all_instances[i] for i in idx]


def list_validation_item_ids(dataset, n_windows: int = 3) -> list[dict]:
    """Return one entry per original validation series eligible for per-series evolution.

    Each entry is {"idx": int, "item_id": str, "n_windows": int, "length": int}.
    Series too short to produce at least one rolling window are skipped.
    """
    pred_len = dataset.prediction_length
    out = []
    for i, entry in enumerate(dataset.validation_dataset):
        target = np.asarray(entry["target"], dtype=np.float64)
        min_history = max(pred_len, 12)
        if len(target) < min_history + pred_len:
            continue
        actual_windows = min(n_windows, (len(target) - min_history) // pred_len)
        if actual_windows <= 0:
            continue
        out.append({
            "idx": i,
            "item_id": str(entry.get("item_id", i)),
            "n_windows": int(actual_windows),
            "length": int(len(target)),
        })
    return out


def get_eval_series_for_item(
    dataset, item_id: str, n_windows: int = 3,
) -> list[dict]:
    """Return rolling-window eval instances for a single original series.

    Mirrors get_eval_series but filters to a single item_id. Each instance
    has the same shape: {input_target, label_target, item_id}.
    """
    pred_len = dataset.prediction_length
    target_id = str(item_id)
    out = []
    for i, entry in enumerate(dataset.validation_dataset):
        entry_id = str(entry.get("item_id", i))
        if entry_id != target_id:
            continue
        target = np.asarray(entry["target"], dtype=np.float64)
        min_history = max(pred_len, 12)
        if len(target) < min_history + pred_len:
            continue
        actual_windows = min(n_windows, (len(target) - min_history) // pred_len)
        for w in range(actual_windows):
            end = len(target) - w * pred_len
            label_start = end - pred_len
            if label_start < min_history:
                break
            out.append({
                "input_target": target[:label_start],
                "label_target": target[label_start:end],
                "item_id": f"{entry_id}_w{w}",
            })
        return out  # one matching entry expected
    return out


def get_test_series_grouped_by_item(dataset) -> dict[str, list[dict]]:
    """Group official test-split rolling-window instances by original item_id.

    Returns a mapping item_id -> list[{input_target, label_target, item_id}].
    """
    grouped: dict[str, list[dict]] = {}
    for inp, label in dataset.test_data:
        item_id = str(inp.get("item_id", len(grouped)))
        grouped.setdefault(item_id, []).append({
            "input_target": np.asarray(inp["target"], dtype=np.float64),
            "label_target": np.asarray(label["target"], dtype=np.float64),
            "item_id": item_id,
        })
    return grouped


_SAFE_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def sanitize_item_id(s: str) -> str:
    """Make an item_id safe for use as a directory name."""
    s = str(s)
    out = []
    for c in s:
        out.append(c if c in _SAFE_ID_CHARS else "_")
    safe = "".join(out).strip("._") or "series"
    return safe[:128]


def get_training_samples(dataset, n: int = 10, seed: int = 42) -> list[np.ndarray]:
    """Return a few training series for validity checks (Stage 1)."""
    samples = []
    for entry in dataset.training_dataset:
        samples.append(np.asarray(entry["target"], dtype=np.float64))
        if len(samples) >= n * 3:
            break

    if len(samples) <= n:
        return samples

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(samples), size=n, replace=False)
    return [samples[i] for i in idx]


# ── Meta dict computation ───────────────────────────────────────────────────

def compute_meta(
    y_hist: np.ndarray,
    prediction_length: int,
    seasonal_period: int,
) -> dict:
    """Compute metadata from a historical target window.

    All computations are O(n) and use only the provided history.
    """
    y = y_hist[np.isfinite(y_hist)]

    if len(y) == 0:
        return {
            "prediction_length": prediction_length,
            "seasonal_period": seasonal_period,
            "length": len(y_hist),
            "n_valid": 0,
        }

    ymean = float(np.mean(y))
    ystd = float(np.std(y))
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    # Trend strength: R^2 of linear fit
    if len(y) >= 3:
        t = np.arange(len(y), dtype=np.float64)
        slope, intercept = np.polyfit(t, y, 1)
        fitted = slope * t + intercept
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - ymean) ** 2)
        trend_strength = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        trend_strength = 0.0

    # Recent window stats (last seasonal_period observations)
    recent = y[-seasonal_period:] if len(y) >= seasonal_period else y
    recent_mean = float(np.mean(recent))
    recent_std = float(np.std(recent))

    # Skewness
    if len(y) >= 3 and ystd > 0:
        skewness = float(sp_stats.skew(y, bias=False))
    else:
        skewness = 0.0

    # Coefficient of variation
    cv = float(ystd / abs(ymean)) if abs(ymean) > 1e-10 else 0.0

    return {
        "prediction_length": prediction_length,
        "seasonal_period": seasonal_period,
        "length": len(y_hist),
        "n_valid": len(y),
        "mean": ymean,
        "std": ystd,
        "median": float(np.median(y)),
        "min_val": ymin,
        "max_val": ymax,
        "range": float(ymax - ymin),
        "positive_frac": float(np.mean(y > 0)),
        "zero_frac": float(np.mean(y == 0)),
        "skewness": skewness,
        "cv": cv,
        "trend_strength": trend_strength,
        "recent_mean": recent_mean,
        "recent_std": recent_std,
    }
