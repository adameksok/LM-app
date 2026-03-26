"""
Model outputs engine — extracts attributes, calculates metrics, builds equations.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from core.plugin_interface import PluginConfig, OutputConfig, MetricConfig


# =========================================================================
# ATTRIBUTE EXTRACTION
# =========================================================================

def get_model_outputs(model, feature_names: List[str], config: PluginConfig) -> Dict[str, Any]:
    """
    Extracts model attributes after fit() and formats them for display.
    """
    outputs = {}

    POSSIBLE_ATTRS = [
        'coef_', 'intercept_', 'classes_', 'n_samples_fit_',
        'class_prior_', 'theta_', 'var_', 'support_vectors_',
        'n_support_', 'components_', 'explained_variance_ratio_',
        'n_iter_', 'n_features_in_', 'feature_names_in_',
        'class_count_', 'n_components_', 'explained_variance_',
        'mean_', 'cluster_centers_', 'labels_', 'inertia_'
    ]

    for attr in POSSIBLE_ATTRS:
        if not hasattr(model, attr):
            continue

        value = getattr(model, attr)
        output_cfg = config.get_output(attr)

        if output_cfg and not output_cfg.show:
            continue

        label = output_cfg.label if output_cfg else attr.rstrip("_").replace("_", " ").title()
        fmt = output_cfg.format if output_cfg else "text"
        hint = output_cfg.hint if output_cfg else ""
        out_type = output_cfg.type if output_cfg else _detect_type(value)

        outputs[attr] = {
            'value': value,
            'label': label,
            'type': out_type,
            'format': fmt,
            'hint': hint,
        }

    return outputs


def _detect_type(value) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "scalar"
    elif isinstance(value, np.ndarray):
        if value.ndim == 1:
            return "vector"
        elif value.ndim == 2:
            return "matrix"
    return "scalar"


# =========================================================================
# METRIC CALCULATION
# =========================================================================

def calculate_metrics(model, X, y, task: str, config: PluginConfig) -> Dict[str, Any]:
    """
    Calculates quality metrics. Returns dict of {metric_id: {value, label, format, hint, good_value}}.
    """
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        accuracy_score, precision_score, recall_score, f1_score,
        silhouette_score
    )

    raw = {}

    if task == "regression":
        y_pred = model.predict(X)
        raw["r2"] = r2_score(y, y_pred)
        raw["mse"] = mean_squared_error(y, y_pred)
        raw["rmse"] = float(np.sqrt(raw["mse"]))
        raw["mae"] = mean_absolute_error(y, y_pred)

    elif task == "classification":
        y_pred = model.predict(X)
        n_classes = len(np.unique(y))
        # Use "binary" for 2 classes (distinct precision/recall/f1),
        # "macro" for multi-class (unweighted mean per class)
        avg = "binary" if n_classes == 2 else "macro"
        raw["accuracy"] = accuracy_score(y, y_pred)
        raw["precision"] = precision_score(y, y_pred, average=avg, zero_division=0)
        raw["recall"] = recall_score(y, y_pred, average=avg, zero_division=0)
        raw["f1"] = f1_score(y, y_pred, average=avg, zero_division=0)
        if hasattr(model, "predict_proba") and n_classes == 2:
            from sklearn.metrics import roc_auc_score
            try:
                y_proba = model.predict_proba(X)[:, 1]
                raw["roc_auc"] = roc_auc_score(y, y_proba)
            except Exception:
                pass

    elif task == "clustering":
        y_pred = model.fit_predict(X) if not hasattr(model, "labels_") else model.labels_
        if len(np.unique(y_pred)) > 1:
            raw["silhouette"] = silhouette_score(X, y_pred)

    elif task == "dimensionality_reduction":
        if hasattr(model, "explained_variance_ratio_"):
            evr = model.explained_variance_ratio_
            raw["explained_variance"] = float(evr[0]) if len(evr) > 0 else 0
            raw["cumulative_variance"] = float(np.sum(evr))

    # Map to display format using plugin config
    result = {}
    for metric_id, value in raw.items():
        mcfg = config.get_metric(metric_id)
        if mcfg and not mcfg.show:
            continue
        result[metric_id] = {
            'value': value,
            'label': mcfg.label if mcfg else metric_id.upper(),
            'format': mcfg.format if mcfg else "decimal",
            'hint': mcfg.hint if mcfg else "",
            'good_value': mcfg.good_value if mcfg else None,
        }

    return result


# =========================================================================
# EQUATION BUILDER
# =========================================================================

def _format_latex(name: str) -> str:
    """Formats feature names for LaTeX, handling multiple underscores as subscripts."""
    parts = name.split("_")
    res = rf"\text{{{parts[0]}}}"
    for p in parts[1:]:
        res = rf"{res}_{{\text{{{p}}}}}"
    return res


def build_equation(model, feature_names: List[str], task: str) -> Optional[str]:
    if not hasattr(model, 'coef_'):
        return None

    coef = model.coef_
    intercept = model.intercept_ if hasattr(model, 'intercept_') else 0

    if hasattr(coef, 'shape') and len(coef.shape) > 1:
        coef = coef[0]
        intercept = intercept[0] if hasattr(intercept, '__len__') else intercept

    terms = []
    for i, (c, name) in enumerate(zip(coef, feature_names)):
        sign = "+" if c >= 0 and i > 0 else "-" if c < 0 else ""
        val = abs(c)
        formatted_name = _format_latex(name)
        terms.append(rf"{sign} {val:.3f} \cdot {formatted_name}")

    intercept_val = abs(intercept)
    intercept_sign = "+" if intercept >= 0 else "-"
    intercept_str = f"{intercept_sign} {intercept_val:.3f}"

    if task == "regression":
        return rf"\hat{{y}} = {' '.join(terms)} {intercept_str}"
    elif task == "classification":
        return rf"\text{{logit}}(p) = {' '.join(terms)} {intercept_str}"
    return None
