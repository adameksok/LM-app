"""
Plugin parser — extended with @output, @metric, @visualization support.
"""

import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.plugin_interface import (
    PluginConfig, PluginMetadata, ParameterConfig, UIConfig,
    OutputConfig, MetricConfig, VisualizationConfig
)


# =========================================================================
# STAGE 1 & 2: Load file and extract docstring
# =========================================================================

def _extract_docstring(content: str) -> Tuple[str, str]:
    pattern = r'"""(.*?)"""'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return "", content
    return match.group(1).strip(), content[match.end():].strip()


# =========================================================================
# STAGE 3: Parse @tags (extended for @output, @metric, @visualization)
# =========================================================================

def _parse_tags(docstring: str) -> Tuple[Dict[str, str], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Returns (header_tags, param_blocks, output_blocks, metric_blocks, viz_blocks).
    """
    header_tags: Dict[str, str] = {}
    param_blocks: List[Dict[str, str]] = []
    output_blocks: List[Dict[str, str]] = []
    metric_blocks: List[Dict[str, str]] = []
    viz_blocks: List[Dict[str, str]] = []

    current_block: Optional[Dict[str, str]] = None
    current_block_type: Optional[str] = None
    last_tag: Optional[str] = None
    last_target: Optional[dict] = None

    for line in docstring.split("\n"):
        stripped = line.strip()

        if not stripped or (not stripped.startswith("@") and not line.startswith(" ") and not line.startswith("\t")):
            continue

        if stripped.startswith("@"):
            colon_pos = stripped.find(":")
            if colon_pos == -1:
                continue
            tag = stripped[1:colon_pos].strip().lower()
            value = stripped[colon_pos + 1:].strip()

            if tag == "param":
                current_block = {"name": value}
                current_block_type = "param"
                param_blocks.append(current_block)
                last_tag = "name"
                last_target = current_block

            elif tag == "output":
                current_block = {"name": value}
                current_block_type = "output"
                output_blocks.append(current_block)
                last_tag = "name"
                last_target = current_block

            elif tag == "metric":
                current_block = {"name": value}
                current_block_type = "metric"
                metric_blocks.append(current_block)
                last_tag = "name"
                last_target = current_block

            elif tag == "visualization":
                current_block = {"name": value}
                current_block_type = "viz"
                viz_blocks.append(current_block)
                last_tag = "name"
                last_target = current_block

            elif current_block is not None:
                # Strip prefix for sub-tags: output_label -> label, metric_show -> show, viz_position -> position
                clean_tag = tag
                for prefix in ("output_", "metric_", "viz_"):
                    if tag.startswith(prefix):
                        clean_tag = tag[len(prefix):]
                        break
                current_block[clean_tag] = value
                last_tag = clean_tag
                last_target = current_block

            else:
                header_tags[tag] = value
                last_tag = tag
                last_target = header_tags

        elif (line.startswith(" ") or line.startswith("\t")):
            if last_tag and last_target:
                last_target[last_tag] += " " + stripped

    return header_tags, param_blocks, output_blocks, metric_blocks, viz_blocks


# =========================================================================
# STAGE 4: Execute code and get model variable
# =========================================================================

def _execute_code(filepath: Path) -> Any:
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "model", None)


# =========================================================================
# STAGE 5: Introspect model.get_params()
# =========================================================================

def _introspect_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "get_params"):
        return model.get_params()
    return {}


# =========================================================================
# STAGE 6: Heuristics
# =========================================================================

def _auto_label(name: str) -> str:
    return name.replace("_", " ").title()


def _auto_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "select"
    return "float"


def _auto_range(param_type: str, default_val: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if param_type == "int":
        d = int(default_val) if default_val is not None else 0
        if d > 0:
            return 1, d * 5, 1
        else:
            return 0, 10, 1
    elif param_type == "float":
        d = float(default_val) if default_val is not None else 0.0
        if 0 < d < 1:
            return 0.001, 1.0, 0.001
        elif d > 0:
            return round(d / 10, 6), round(d * 10, 6), round(d / 100, 6)
        else:
            return 0.0, 1.0, 0.01
    return None, None, None


def _build_parameters(param_blocks, sklearn_params):
    user_defined = {pb["name"]: pb for pb in param_blocks}
    result = []

    for param_name, default_val in sklearn_params.items():
        if default_val is None:
            continue
        user_block = user_defined.get(param_name, {})
        show = user_block.get("show", "true").lower() != "false"
        if not show:
            continue
        detected_type = _auto_type(default_val)
        param_type = user_block.get("type", detected_type)
        if param_type == "select" and "options" not in user_block:
            continue

        label = user_block.get("label", _auto_label(param_name))
        hint = user_block.get("hint", "")

        if param_type in ("int", "float"):
            auto_min, auto_max, auto_step = _auto_range(param_type, default_val)
            min_val = float(user_block["min"]) if "min" in user_block else auto_min
            max_val = float(user_block["max"]) if "max" in user_block else auto_max
            step_raw = user_block.get("step", None)
            step = auto_step if step_raw in (None, "log") else float(step_raw)
            default = float(user_block["default"]) if "default" in user_block else float(default_val)
            if param_type == "int":
                min_val, max_val, step, default = int(min_val or 0), int(max_val or 10), int(step or 1), int(default)
            result.append(ParameterConfig(name=param_name, label=label, type=param_type,
                                          min_val=min_val, max_val=max_val, step=step,
                                          default=default, hint=hint, show=True))
        elif param_type == "bool":
            default = user_block.get("default", str(default_val)).lower() == "true"
            result.append(ParameterConfig(name=param_name, label=label, type="bool",
                                          default=default, hint=hint, show=True))
        elif param_type == "select":
            options = [o.strip() for o in user_block.get("options", "").split(",")]
            default = user_block.get("default", str(default_val))
            result.append(ParameterConfig(name=param_name, label=label, type="select",
                                          options=options, default=default, hint=hint, show=True))
    return result


def _build_outputs(output_blocks: List[Dict[str, str]]) -> List[OutputConfig]:
    result = []
    for block in output_blocks:
        show = block.get("show", "true").lower() != "false"
        result.append(OutputConfig(
            name=block["name"],
            label=block.get("label", _auto_label(block["name"].rstrip("_"))),
            type=block.get("type", "scalar"),
            show=show,
            format=block.get("format", "text"),
            hint=block.get("hint", ""),
            condition=block.get("condition", "")
        ))
    return result


def _build_metrics(metric_blocks: List[Dict[str, str]]) -> List[MetricConfig]:
    result = []
    for block in metric_blocks:
        show = block.get("show", "true").lower() != "false"
        good_val = float(block["good_value"]) if "good_value" in block else None
        result.append(MetricConfig(
            name=block["name"],
            label=block.get("label", _auto_label(block["name"])),
            show=show,
            format=block.get("format", "decimal"),
            hint=block.get("hint", ""),
            good_value=good_val
        ))
    return result


def _build_visualizations(viz_blocks: List[Dict[str, str]]) -> List[VisualizationConfig]:
    result = []
    for block in viz_blocks:
        show = block.get("show", "true").lower() != "false"
        result.append(VisualizationConfig(
            name=block["name"],
            label=block.get("label", _auto_label(block["name"])),
            show=show,
            position=block.get("position", "side")
        ))
    return result


# =========================================================================
# STAGE 7: Map @task to UI config (defaults when no tags defined)
# =========================================================================

TASK_UI_MAP = {
    "classification": UIConfig(chart_type="decision_boundary",
                               metrics=["Accuracy", "F1-Score"], extras=["confusion_matrix"]),
    "regression": UIConfig(chart_type="regression_fit",
                           metrics=["R²", "RMSE"], extras=["residuals"]),
    "clustering": UIConfig(chart_type="cluster_scatter",
                           metrics=["Silhouette"], extras=["elbow"]),
    "dimensionality_reduction": UIConfig(chart_type="projection_2d",
                                         metrics=["Explained Variance"], extras=["variance_bars"]),
}

DEFAULT_METRICS = {
    "regression": [
        MetricConfig("r2", "R²", True, "percent", "", 0.7),
        MetricConfig("rmse", "RMSE", True, "decimal"),
        MetricConfig("mae", "MAE", True, "decimal"),
    ],
    "classification": [
        MetricConfig("accuracy", "Accuracy", True, "percent", "", 0.8),
        MetricConfig("precision", "Precision", True, "percent", "", 0.7),
        MetricConfig("recall", "Recall", True, "percent", "", 0.7),
        MetricConfig("f1", "F1-Score", True, "percent", "", 0.7),
    ],
    "clustering": [
        MetricConfig("silhouette", "Silhouette", True, "decimal"),
    ],
    "dimensionality_reduction": [
        MetricConfig("explained_variance", "Explained Variance", True, "percent"),
        MetricConfig("cumulative_variance", "Cumulative Variance", True, "percent", "", 0.9),
    ],
}

DEFAULT_VISUALIZATIONS = {
    "regression": [
        VisualizationConfig("equation", "Wyuczone równanie", True, "top"),
        VisualizationConfig("regression_fit", "Dopasowanie modelu", True, "main"),
    ],
    "classification": [
        VisualizationConfig("decision_boundary", "Decision Boundary", True, "main"),
        VisualizationConfig("confusion_matrix", "Confusion Matrix", True, "side"),
    ],
    "clustering": [
        VisualizationConfig("cluster_centers_overlay", "Cluster Centers", True, "main"),
    ],
    "dimensionality_reduction": [
        VisualizationConfig("variance_bar", "Variance", True, "side"),
        VisualizationConfig("projection_2d", "2D Projection", True, "main"),
    ],
}


# =========================================================================
# PUBLIC API
# =========================================================================

def parse_plugin_file(filepath: Path) -> Optional[PluginConfig]:
    content = filepath.read_text(encoding="utf-8")
    docstring, _ = _extract_docstring(content)
    if not docstring:
        return None

    header_tags, param_blocks, output_blocks, metric_blocks, viz_blocks = _parse_tags(docstring)

    if "model" not in header_tags or "task" not in header_tags:
        return None

    model_instance = _execute_code(filepath)
    if model_instance is None:
        return None

    sklearn_params = _introspect_model(model_instance)
    parameters = _build_parameters(param_blocks, sklearn_params)

    # Build output/metric/viz configs (use defaults if not defined)
    task = header_tags.get("task", "classification")
    outputs = _build_outputs(output_blocks)
    metrics = _build_metrics(metric_blocks) if metric_blocks else DEFAULT_METRICS.get(task, [])
    visualizations = _build_visualizations(viz_blocks) if viz_blocks else DEFAULT_VISUALIZATIONS.get(task, [])

    ui_config = TASK_UI_MAP.get(task, TASK_UI_MAP["classification"])

    metadata = PluginMetadata(
        model_class=header_tags.get("model", ""),
        task=task,
        name=header_tags.get("name", _auto_label(filepath.stem)),
        description=header_tags.get("description", ""),
        icon=header_tags.get("icon", "")
    )

    return PluginConfig(
        metadata=metadata,
        parameters=parameters,
        outputs=outputs,
        metrics=metrics,
        visualizations=visualizations,
        ui_config=ui_config,
        model_instance=model_instance,
        source_file=str(filepath)
    )
