import streamlit as st
from core.plugin_interface import ParameterConfig
from typing import List, Dict, Any


def render_sidebar(parameters: List[ParameterConfig]) -> Dict[str, Any]:
    """
    Renders dynamic controls in the sidebar based on parsed ParameterConfig.
    Returns dict of {param_name: current_value}.
    """
    params = {}

    st.sidebar.header("⚙️ Model Parameters")

    if not parameters:
        st.sidebar.info("No configurable parameters for this model.")
        return params

    for param in parameters:
        if not param.show:
            continue

        if param.type == "int":
            params[param.name] = st.sidebar.slider(
                label=param.label,
                min_value=int(param.min_val),
                max_value=int(param.max_val),
                value=int(param.default),
                step=int(param.step) if param.step else 1,
                help=param.hint
            )

        elif param.type == "float":
            params[param.name] = st.sidebar.slider(
                label=param.label,
                min_value=float(param.min_val),
                max_value=float(param.max_val),
                value=float(param.default),
                step=float(param.step) if param.step else 0.01,
                help=param.hint
            )

        elif param.type == "bool":
            params[param.name] = st.sidebar.checkbox(
                label=param.label,
                value=bool(param.default),
                help=param.hint
            )

        elif param.type == "select":
            options = param.options or []
            default_idx = options.index(param.default) if param.default in options else 0
            params[param.name] = st.sidebar.selectbox(
                label=param.label,
                options=options,
                index=default_idx,
                help=param.hint
            )

    return params
