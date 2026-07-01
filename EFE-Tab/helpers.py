"""Depth-3 LightGBM single-tree, a decision tree, helper with Optuna HPO."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _lgbm_tree_to_text(
    node: dict, feature_names: list[str], categorical_maps: dict[str, list],
) -> str:
    """Render a LightGBM tree dict (from booster_.dump_model()['tree_info'][i]
    ['tree_structure']) as indented text in sklearn's `export_text` style.

    Numeric splits → `feat <= threshold`.
    Categorical splits → `feat in {cat_a, cat_b, ...}`.
    Leaves → `class: 0/1  (prob=...)`.
    """
    def walk(n: dict, depth: int) -> list[str]:
        prefix = "|   " * depth + "|--- "
        if "leaf_value" in n:
            leaf = float(n["leaf_value"])
            prob = 1.0 / (1.0 + np.exp(-leaf))
            cls = 1 if leaf > 0 else 0
            return [f"{prefix}class: {cls}  (prob={prob:.3f})"]
        feat = feature_names[int(n["split_feature"])]
        dtype = n.get("decision_type", "<=")
        thresh = n["threshold"]
        if dtype == "==":
            # Categorical split: threshold is "i||j||k" — indices into cat list
            try:
                cats = categorical_maps.get(feat, [])
                idx = [int(x) for x in str(thresh).split("||")]
                names = [str(cats[i]) for i in idx if 0 <= i < len(cats)]
                left = f"{feat} in {{{', '.join(names)}}}"
                right = f"{feat} not in {{{', '.join(names)}}}"
            except Exception:
                left = f"{feat} == {thresh}"
                right = f"{feat} != {thresh}"
        else:
            try:
                t = float(thresh)
                left = f"{feat} <= {t:.4f}"
                right = f"{feat} >  {t:.4f}"
            except (TypeError, ValueError):
                left = f"{feat} {dtype} {thresh}"
                right = f"{feat} not {dtype} {thresh}"
        out = [f"{prefix}{left}"]
        out.extend(walk(n["left_child"], depth + 1))
        out.append(f"{prefix}{right}")
        out.extend(walk(n["right_child"], depth + 1))
        return out

    return "\n".join(walk(node, 0))


def _prepare_for_lgbm(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list]]:
    """Convert object columns to pandas `category` dtype so LightGBM picks
    them up via its `auto` categorical-feature detection.

    Returns (X_prepared, categorical_cols, category_maps) where
    `category_maps[col]` is the ordered list of category values — needed to
    decode LightGBM's categorical split thresholds (given as indices) back
    into category names.
    """
    X = X.copy()
    cat_cols: list[str] = []
    cat_maps: dict[str, list] = {}
    for col in X.columns:
        dt = X[col].dtype
        if dt == object or str(dt) == "category":
            X[col] = X[col].astype("category")
            cat_cols.append(col)
            cat_maps[col] = list(X[col].cat.categories)
    return X, cat_cols, cat_maps


def _fit_depth3_lgbm_tree_text(
    X_out: pd.DataFrame,
    y: pd.Series,
    original_cols: list[str],
    n_trials: int = 20,
    cv_folds: int = 3,
    seed: int = 42,
) -> str:
    """Optuna HPO for a depth-3 LightGBM single tree, refit on full data,
    return annotated text. Categorical columns are split natively (Fisher
    optimal category-set splits) instead of being one-hot encoded or dropped.

    Fixed tree shape: `n_estimators=1, max_depth=3, num_leaves=8,
    learning_rate=1.0`. HPO tunes {min_data_in_leaf, min_gain_to_split,
    feature_fraction, class_weight}.
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold

    try:
        import lightgbm as lgb
    except ImportError:
        return "LightGBM tree unavailable: lightgbm not installed."

    try:
        X_prep, cat_cols, cat_maps = _prepare_for_lgbm(X_out)
        feature_names = X_prep.columns.tolist()
        if not feature_names:
            return "No features for LightGBM decision tree."

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_data = [
            (X_prep.iloc[tr], y.iloc[tr], X_prep.iloc[va], y.iloc[va])
            for tr, va in skf.split(X_prep, y)
        ]

        fixed = dict(
            n_estimators=1, max_depth=3, num_leaves=8,
            learning_rate=1.0, objective="binary",
            random_state=seed, verbose=-1, n_jobs=1,
        )

        def objective(trial):
            params = {
                **fixed,
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
                "min_gain_to_split": trial.suggest_float(
                    "min_gain_to_split", 1e-6, 0.1, log=True,
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.7, 1.0,
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced"],
                ),
            }
            scores = []
            for X_tr, y_tr, X_va, y_va in fold_data:
                clf = lgb.LGBMClassifier(**params)
                clf.fit(X_tr, y_tr, categorical_feature=cat_cols or "auto")
                proba = clf.predict_proba(X_va)[:, 1]
                scores.append(roc_auc_score(y_va, proba))
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

        best = {**fixed, **study.best_params}
        clf = lgb.LGBMClassifier(**best)
        clf.fit(X_prep, y, categorical_feature=cat_cols or "auto")

        # Dump the single tree and render as text
        model_json = clf.booster_.dump_model()
        tree_node = model_json["tree_info"][0]["tree_structure"]
        # LightGBM's dump feature_names is authoritative
        dump_feat_names = model_json.get("feature_names", feature_names)
        tree_text = _lgbm_tree_to_text(tree_node, dump_feat_names, cat_maps)

        generated_cols = [c for c in feature_names if c not in original_cols]
        used_in_tree = [c for c in feature_names if c in tree_text]
        lines = [
            f"Depth-3 LightGBM tree (HPO {n_trials} trials, CV {cv_folds}-fold, "
            f"best CV AUC={study.best_value:.4f}):",
            f"Categorical features (native splits): "
            f"{cat_cols if cat_cols else '(none)'}",
            "Feature legend:",
        ]
        for col in used_in_tree:
            tag = "[GENERATED]" if col in generated_cols else "[ORIGINAL]"
            lines.append(f"  {col} {tag}")
        lines.append("")
        lines.append(tree_text)
        return "\n".join(lines)
    except Exception as e:
        return f"Depth-3 LightGBM tree failed: {e}"
