"""Fixed v3 classical scores and deterministic sklearn model factories."""
from __future__ import annotations

from functools import partial
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from nr_v3_data import FEATURES

CLASSICAL = ("correlation", "sss", "dmrs_coherence", "llr_abs", "dmrs_fit", "joint_pilot")
_INDEX = {name: i for i, name in enumerate(FEATURES)}
_LOGGED = {"peak_ratio", "abs_cfo", "dmrs_power", "dmrs_variation", "llr_abs", "estimated_noise"}


def _log_transform(values: np.ndarray, log_columns: tuple[int, ...]) -> np.ndarray:
    """Pickleable feature transform used in persisted sklearn pipelines."""
    values = np.asarray(values, dtype=float).copy()
    if log_columns:
        values[:, log_columns] = np.log1p(np.clip(values[:, log_columns], 0, None))
    return values


def classical_scores(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or name not in CLASSICAL:
        raise ValueError("invalid classic score input")
    if name == "dmrs_fit": return -x[:, _INDEX["dmrs_fit_error"]]
    if name == "joint_pilot": return x[:, _INDEX["sss"]] * np.sqrt(np.clip(x[:, _INDEX["dmrs_coherence"]], 0, None))
    return x[:, _INDEX[name]]


def model_specs() -> list[dict]:
    specs = []
    for c in (.01, .1, 1):
        for weight in (None, "balanced"):
            specs.append({"name": f"logistic_C{c}_{weight or 'none'}", "family": "logistic", "params": {"C": c, "class_weight": weight}})
    for leaves in (7, 15, 31):
        for lr in (.03, .1): specs.append({"name": f"hgb_l{leaves}_lr{lr}", "family": "hgb", "params": {"max_leaf_nodes": leaves, "learning_rate": lr}})
    for depth in (6, 12, None):
        for leaf in (5, 20): specs.append({"name": f"rf_d{depth}_l{leaf}", "family": "rf", "params": {"max_depth": depth, "min_samples_leaf": leaf}})
    for hidden in ((16,), (32,), (32, 16)):
        for alpha in (.0001, .01): specs.append({"name": f"mlp_{'-'.join(map(str, hidden))}_a{alpha}", "family": "mlp", "params": {"hidden_layer_sizes": hidden, "alpha": alpha}})
    return specs


def make_model(spec: dict, seed: int, keep_features: tuple[str, ...] = FEATURES):
    if spec not in model_specs() or not keep_features or any(name not in _INDEX for name in keep_features):
        raise ValueError("unknown model specification or feature")
    columns = [_INDEX[name] for name in keep_features]
    log_columns = tuple(i for i, name in enumerate(keep_features) if name in _LOGGED)
    transform = partial(_log_transform, log_columns=log_columns)
    preprocess_steps = [("select_log", Pipeline([("log", FunctionTransformer(transform, validate=False))]), columns)]
    preprocessor = ColumnTransformer(preprocess_steps, remainder="drop")
    family, params = spec["family"], spec["params"]
    if family == "logistic": estimator = LogisticRegression(**params, max_iter=2000, random_state=seed)
    elif family == "hgb": estimator = HistGradientBoostingClassifier(**params, max_iter=200, min_samples_leaf=30, l2_regularization=1, early_stopping=False, random_state=seed)
    elif family == "rf": estimator = RandomForestClassifier(**params, n_estimators=300, max_features="sqrt", class_weight="balanced_subsample", n_jobs=1, random_state=seed)
    elif family == "mlp": estimator = MLPClassifier(**params, activation="relu", solver="adam", learning_rate_init=.001, batch_size=256, max_iter=300, early_stopping=False, random_state=seed)
    else: raise ValueError("unknown model family")
    steps = [("features", preprocessor)]
    if family in ("logistic", "mlp"): steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)
