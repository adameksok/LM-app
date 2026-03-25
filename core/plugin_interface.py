from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ParameterConfig:
    """Configuration for a single model parameter (rendered as a UI control)."""
    name: str
    label: str = ""
    type: str = "float"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[Any] = None
    default: Any = None
    options: Optional[List[Any]] = None
    option_labels: Optional[Dict[Any, str]] = None
    hint: str = ""
    show: bool = True


@dataclass
class OutputConfig:
    """Configuration for a model attribute displayed after fit()."""
    name: str            # sklearn attr name, e.g. "coef_"
    label: str = ""
    type: str = "scalar"  # scalar, vector, matrix, array_of_vectors, labels, integer
    show: bool = True
    format: str = "text"  # text, bar_chart, table, heatmap, equation, percentage_bar, scatter_overlay
    hint: str = ""
    condition: str = ""   # e.g. "kernel == 'linear'"


@dataclass
class MetricConfig:
    """Configuration for a quality metric."""
    name: str            # metric identifier, e.g. "r2", "accuracy"
    label: str = ""
    show: bool = True
    format: str = "decimal"  # decimal, percent, integer
    hint: str = ""
    good_value: Optional[float] = None


@dataclass
class VisualizationConfig:
    """Configuration for an extra visualization."""
    name: str            # viz identifier, e.g. "equation", "confusion_matrix"
    label: str = ""
    show: bool = True
    position: str = "side"  # main, side, bottom, top


@dataclass
class PluginMetadata:
    """Descriptive metadata parsed from @tags in the docstring."""
    model_class: str = ""
    task: str = ""
    name: str = ""
    description: str = ""
    icon: str = ""


@dataclass
class UIConfig:
    """UI configuration auto-derived from @task."""
    chart_type: str = ""
    metrics: List[str] = field(default_factory=list)
    extras: List[str] = field(default_factory=list)


@dataclass
class PluginConfig:
    """Complete plugin configuration — the output of the parser."""
    metadata: PluginMetadata = field(default_factory=PluginMetadata)
    parameters: List[ParameterConfig] = field(default_factory=list)
    outputs: List[OutputConfig] = field(default_factory=list)
    metrics: List[MetricConfig] = field(default_factory=list)
    visualizations: List[VisualizationConfig] = field(default_factory=list)
    ui_config: UIConfig = field(default_factory=UIConfig)
    model_instance: Any = None
    source_file: str = ""

    def get_output(self, attr_name: str) -> Optional[OutputConfig]:
        for o in self.outputs:
            if o.name == attr_name:
                return o
        return None

    def get_metric(self, metric_id: str) -> Optional[MetricConfig]:
        for m in self.metrics:
            if m.name == metric_id:
                return m
        return None

    def get_visualization(self, viz_id: str) -> Optional[VisualizationConfig]:
        for v in self.visualizations:
            if v.name == viz_id:
                return v
        return None
