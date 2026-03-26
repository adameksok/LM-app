"""
Rich results panel — renders model outputs, metrics, and visualizations.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from typing import Dict, Any, List, Optional

from core.plugin_interface import PluginConfig, VisualizationConfig
from core.model_outputs import get_model_outputs, calculate_metrics, build_equation
from core.i18n_utils import t


def _inject_equation_styles():
    """Injects global CSS for the Learned Equation widget."""
    st.markdown(f"""
        <style>
        .eq-header-bar {{ 
            background:#004b87; color:white; padding:8px 16px; border-radius:8px 8px 0 0; 
            font-size:11px; font-weight:700; letter-spacing:1.2px; text-align:left; 
            display:flex; align-items:center; gap:8px; margin-bottom: 0px; 
        }}
        .eq-footer-row {{ 
            display:flex; justify-content:space-between; color:#64748b; 
            font-size:10px; font-family:monospace; text-transform:uppercase; 
            letter-spacing:0.5px; margin-top: 10px; border-top: 1px solid #f1f5f9; padding-top: 8px;
        }}
        /* Bind header and the following container together by removing gap */
        div[data-testid="stVerticalBlock"] > div:has(div.eq-header-bar) + div {{
            margin-top: -16px !important;
        }}
        /* Ensure KaTeX inside the widget is properly sized */
        .katex {{ font-size: 1.4em !important; }}
        </style>
    """, unsafe_allow_html=True)


def render_data_preview(task: str, X: np.ndarray, y: Optional[np.ndarray], feature_names: List[str]):
    """Renders a scatter plot of the raw data before the model is run."""
    st.markdown(f"#### 🖼️ {t('viz.data_preview_header')}")
    fig = go.Figure()

    if task == "regression":
        X_flat = X.reshape(-1, 1) if X.ndim == 1 else X
        fig.add_trace(go.Scatter(
            x=X_flat[:, 0], y=y, mode='markers', name=t('viz.data'),
            marker=dict(size=8, color='#3498db', opacity=0.7)
        ))
        fig.update_layout(
            title=t("viz.scatter_plot"),
            xaxis_title=feature_names[0] if feature_names else "X",
            yaxis_title="y",
            template='plotly_white'
        )
    elif task == "classification":
        if X.shape[1] >= 2:
            for cls in np.unique(y):
                mask = y == cls
                fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=f"{t('viz.class')} {cls}",
                                         marker=dict(size=8, line=dict(width=1, color='white'))))
            fig.update_layout(title=t("viz.class_dist"), xaxis_title=feature_names[0] if feature_names else "X₁", 
                              yaxis_title=feature_names[1] if len(feature_names)>1 else "X₂", template='plotly_white')
        else:
            st.info("Dane mają mniej niż 2 cechy - podgląd 2D niedostępny.")
            return
    elif task == "clustering":
        if X.shape[1] >= 2:
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', name='Dane',
                                     marker=dict(size=8, color='#95a5a6', opacity=0.7)))
            fig.update_layout(title="Rozkład punktów", xaxis_title=feature_names[0] if feature_names else "X₁", 
                              yaxis_title=feature_names[1] if len(feature_names)>1 else "X₂", template='plotly_white')
        else:
            st.info("Dane mają mniej niż 2 cechy - podgląd 2D niedostępny.")
            return
    else:
        st.info("Podgląd wizualny dostępny po uruchomieniu modelu dla tego zadania.")
        return

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="data_preview")


def fit_model_instance(config: PluginConfig, model, X, y, params):
    """Applies parameters and fits the model based on task type."""
    task = config.metadata.task
    try:
        if params:
            model.set_params(**params)
    except Exception:
        pass

    if task in ("dimensionality_reduction", "clustering"):
        model.fit(X)
    else:
        model.fit(X, y)


def render_results_panel(config: PluginConfig, model, X, y, params, feature_names):
    """Main entry — renders the full results panel after model.fit()."""
    task = config.metadata.task
    
    # Model should be fitted by fit_model_instance before this call for efficiency,
    # but we ensure it here as well for backwards compatibility.
    fit_model_instance(config, model, X, y, params)
        
    # --- MAIN VISUALIZATIONS ---
    visible_vizs = [v for v in config.visualizations if v.show]
    main_vizs = [v for v in visible_vizs if v.position == "main"]
    side_vizs = [v for v in visible_vizs if v.position == "side"]
    bottom_vizs = [v for v in visible_vizs if v.position == "bottom"]

    for viz in main_vizs:
        _render_viz(viz, model, X, y, task, feature_names)

    # --- EQUATION ---
    eq_vizs = [v for v in visible_vizs if v.name == "equation" and v.position == "top"]
    if eq_vizs:
        _inject_equation_styles()
        equation = build_equation(model, feature_names, task)
        if equation:
            # Extract raw values for footer
            coef_raw = model.coef_
            intercept_raw = model.intercept_ if hasattr(model, 'intercept_') else 0
            if hasattr(coef_raw, 'shape') and len(coef_raw.shape) > 1: coef_raw = coef_raw[0]
            if hasattr(intercept_raw, '__len__'): intercept_raw = intercept_raw[0]
            
            if hasattr(coef_raw, '__len__') and len(coef_raw) > 1:
                coef_str = f"{len(coef_raw)} features"
            else:
                c_val = coef_raw[0] if hasattr(coef_raw, '__len__') else coef_raw
                coef_str = f"{c_val:.5f}"

            st.markdown(f'<div class="eq-header-bar">📐 {t("viz.learned_eq").upper()}</div>', unsafe_allow_html=True)

            with st.container(border=True):
                st.latex(equation.replace("·", r" \cdot ").replace("ŷ", r"\hat{y}"))
                st.markdown(f"""
                    <div class="eq-footer-row">
                        <span><b>{t("viz.coef_label")}</b>: {coef_str}</span>
                        <span><b>{t("viz.intercept_label")}</b>: {intercept_raw:.5f}</span>
                    </div>
                """, unsafe_allow_html=True)


    # --- METRICS ---
    # Metrics now full width under equation
    metrics = calculate_metrics(model, X, y, task, config)
    if metrics:
        with st.container(border=True):
            st.markdown(f"**📈 {t('viz.quality_metrics')}**")
            m_cols = st.columns(len(metrics))
            for i, (mid, mdata) in enumerate(metrics.items()):
                col = m_cols[i]
                value = mdata['value']
                label = mdata['label']
                fmt = mdata['format']
                good = mdata.get('good_value')

                if fmt == "percent":
                    display_val = f"{value:.1%}"
                elif fmt == "integer":
                    display_val = str(int(value))
                else:
                    display_val = f"{value:.4e}"

                delta = None
                if good is not None and isinstance(value, (int, float)):
                    delta = f"{'✅ good' if value >= good else '⚠️ low'}"

                col.metric(label=label, value=display_val, delta=delta)
                if mdata.get('hint'):
                    col.caption(mdata['hint'][:120])

    # --- BOTTOM VISUALIZATIONS ---
    # Stacked vertically under metrics
    for viz in bottom_vizs:
        _render_viz(viz, model, X, y, task, feature_names)


# =========================================================================
# OUTPUT FORMATTERS
# =========================================================================

def _render_output(attr_name: str, odata: Dict[str, Any], feature_names: List[str]):
    value = odata['value']
    label = odata['label']
    fmt = odata.get('format', 'text')
    hint = odata.get('hint', '')

    st.markdown(f"**📋 {t(label)}**")
    if fmt == "text":
        if isinstance(value, np.ndarray):
            st.code(str(value))
        else:
            st.write(f"**{value}**")

    elif fmt == "bar_chart":
        if isinstance(value, np.ndarray) and value.ndim == 1:
            names = feature_names[:len(value)] if len(feature_names) >= len(value) else [f"f{i}" for i in range(len(value))]
            fig = go.Figure(go.Bar(x=names, y=value, marker_color=['#e74c3c' if v < 0 else '#3498db' for v in value]))
            fig.update_layout(title="", template='plotly_white', height=250, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"out_bar_{attr_name}")
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            fig = go.Figure(go.Bar(x=[f"f{i}" for i in range(value.shape[1])], y=value[0]))
            fig.update_layout(title="", template='plotly_white', height=250, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"out_bar_2d_{attr_name}")
        else:
            st.write(f"**{value}**")

    elif fmt == "table":
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                df = pd.DataFrame(value, columns=feature_names[:value.shape[1]] if len(feature_names) >= value.shape[1] else None)
                st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(pd.DataFrame({'Value': value}), use_container_width=True)
        else:
            st.write(f"**{value}**")

    elif fmt == "percentage_bar":
        if isinstance(value, np.ndarray):
            for i, v in enumerate(value):
                st.progress(float(v), text=f"PC{i+1}: {v:.1%}")
        else:
            st.progress(float(value), text=f"{value:.1%}")

    elif fmt == "heatmap":
        if isinstance(value, np.ndarray) and value.ndim == 2:
            fig = px.imshow(value, text_auto=".2f", color_continuous_scale='RdBu_r')
            fig.update_layout(title="", height=250, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"out_heat_{attr_name}")

    elif fmt == "scatter_overlay":
        st.caption(f"{len(value)} items")
        st.code(str(value[:5]) + (" ..." if len(value) > 5 else ""))

    else:
        st.write(f"**{value}**")

    if hint:
        st.markdown(f'<div class="ml-alert ml-alert-info"><span>💡</span><div>{hint}</div></div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:12px 0; border:none; border-top:1px solid #f0f0f0'>", unsafe_allow_html=True)


# =========================================================================
# VISUALIZATION RENDERERS
# =========================================================================

def _render_viz(viz: VisualizationConfig, model, X, y, task, feature_names):
    """Dispatches to the correct visualization renderer."""

    viz_map = {
        "decision_boundary": _viz_decision_boundary,
        "regression_fit": _viz_regression_fit,
        "confusion_matrix": _viz_confusion_matrix,
        "equation": lambda *args: None,  # handled separately at top
        "coefficients_bar": _viz_coefficients_bar,
        "coefficients_table": _viz_coefficients_table,
        "residuals_plot": _viz_residuals,
        "cluster_centers_overlay": _viz_cluster_centers,
        "variance_bar": _viz_variance_bar,
        "cumulative_variance_line": _viz_cumulative_variance,
        "projection_2d": _viz_projection_2d,
        "loadings_heatmap": _viz_loadings_heatmap,
        "class_distribution": _viz_class_distribution,
        "roc_curve": _viz_roc_curve,
        "probability_distribution": _viz_probability_distribution,
        "precision_recall_curve": _viz_precision_recall_curve,
        "support_vectors_overlay": _viz_support_vectors,
        "data_table": _viz_data_table,
    }

    renderer = viz_map.get(viz.name)
    if renderer:
        try:
            with st.container(border=True):
                renderer(model, X, y, task, feature_names, viz)
        except Exception as e:
            st.warning(f"⚠️ {viz.label}: {e}")


def _viz_decision_boundary(model, X, y, task, feature_names, viz):
    if X.shape[1] < 2:
        return
    if X.shape[1] > 2:
        st.markdown(f'<div class="ml-alert ml-alert-info"><span>💡</span><div>{t("viz.high_dim_hint")}</div></div>', unsafe_allow_html=True)
        return
        
    fig = go.Figure()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    step = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, step), y=np.arange(y_min, y_max, step), z=Z,
                              showscale=False, opacity=0.3, colorscale='RdBu', hoverinfo='skip'))
    for cls in np.unique(y):
        mask = y == cls
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=f"{t('viz.class')} {cls}",
                                  marker=dict(size=8, line=dict(width=1, color='white'))))
    fig.update_layout(title=t(viz.label), xaxis_title=feature_names[0] if feature_names else "X₁", yaxis_title=feature_names[1] if len(feature_names)>1 else "X₂", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_decision_boundary")


def _viz_regression_fit(model, X, y, task, feature_names, viz):
    """Scatter plot of data points + fitted regression line, OR Actual vs Predicted for >1 feature."""
    outliers = st.session_state.get("current_outliers")
    has_outliers = outliers is not None and len(outliers) == len(X)
        
    if X.shape[1] == 1:
        X_flat = X.reshape(-1, 1)
        fig = go.Figure()
        
        # Data points
        if has_outliers:
            fig.add_trace(go.Scatter(
                x=X_flat[~outliers, 0], y=y[~outliers], mode='markers', name=f"{t('viz.data')} ({t('viz.normal')})",
                marker=dict(size=8, color='#3498db', opacity=0.7)
            ))
            fig.add_trace(go.Scatter(
                x=X_flat[outliers, 0], y=y[outliers], mode='markers', name=t('viz.outliers'),
                marker=dict(size=10, color='#e74c3c', opacity=0.9, symbol='x')
            ))
        else:
            fig.add_trace(go.Scatter(
                x=X_flat[:, 0], y=y, mode='markers', name='Dane',
                marker=dict(size=8, color='#3498db', opacity=0.7)
            ))
        
        # Fitted line (sorted for smooth line)
        x_min, x_max = X_flat.min(), X_flat.max()
        x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        fig.add_trace(go.Scatter(
            x=x_line[:, 0], y=y_line, mode='lines', name=t('viz.fit'),
            line=dict(color='#e74c3c', width=3)
        ))
        
        fig.update_layout(
            title=t(viz.label),
            xaxis_title=feature_names[0] if feature_names else "X",
            yaxis_title="Cel (y)",
            hovermode='closest',
            template='plotly_white'
        )
    else:
        # Multiple regression (N > 1) -> Actual vs Predicted
        y_pred = model.predict(X)
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y, y=y_pred, mode='markers', name='Predykcje',
            marker=dict(size=8, color='#3498db', opacity=0.7)
        ))
        
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Idealne dopasowanie',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{t(viz.label)} ({t('viz.predicted').capitalize()}): {t('viz.actual')} vs {t('viz.predicted')}",
            xaxis_title=f"{t('viz.actual')} (y)",
            yaxis_title=t('viz.predicted'),
            hovermode='closest',
            template='plotly_white'
        )
        
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_regression_fit")


def _viz_confusion_matrix(model, X, y, task, feature_names, viz):
    y_pred = model.predict(X)
    cm = sk_confusion_matrix(y, y_pred)
    labels = sorted(np.unique(y))
    fig = px.imshow(cm, text_auto=True, x=[str(l) for l in labels], y=[str(l) for l in labels],
                    color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(title=t(viz.label), height=350)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_confusion_matrix")


def _viz_coefficients_bar(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'coef_'):
        return
    coef = model.coef_.flatten()[:len(feature_names)]
    fig = go.Figure(go.Bar(x=feature_names[:len(coef)], y=coef,
                           marker_color=['#e74c3c' if v < 0 else '#3498db' for v in coef]))
    fig.update_layout(title=t(viz.label), template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_coefficients_bar")


def _viz_coefficients_table(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'coef_'):
        return
    coef = model.coef_.flatten()[:len(feature_names)]
    df = pd.DataFrame({'Feature': feature_names[:len(coef)], 'Coefficient': coef})
    st.dataframe(df, use_container_width=True)


def _viz_residuals(model, X, y, task, feature_names, viz):
    y_pred = model.predict(X)
    residuals = y - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(size=6, color='#3498db')))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title=t(viz.label), xaxis_title=t("viz.predicted"), yaxis_title=t("viz.residual"), template='plotly_white')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_residuals")


def _viz_cluster_centers(model, X, y, task, feature_names, viz):
    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
    fig = go.Figure()
    for cls in np.unique(labels):
        mask = labels == cls
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=f'Cluster {cls}',
                                  marker=dict(size=8)))
    if hasattr(model, 'cluster_centers_'):
        c = model.cluster_centers_
        fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode='markers', name='Centroids',
                                  marker=dict(size=16, color='black', symbol='x')))
    fig.update_layout(title=t(viz.label), template='plotly_white')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_cluster_centers")


def _viz_variance_bar(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'explained_variance_ratio_'):
        return
    evr = model.explained_variance_ratio_
    fig = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(evr))], y=evr, marker_color='#3498db'))
    fig.update_layout(title=t(viz.label), yaxis_title="Variance Ratio", template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_variance_bar")


def _viz_cumulative_variance(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'explained_variance_ratio_'):
        return
    evr = model.explained_variance_ratio_
    cumul = np.cumsum(evr)
    fig = go.Figure(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumul))], y=cumul, mode='lines+markers',
                                marker=dict(size=8, color='#e74c3c')))
    fig.add_hline(y=0.9, line_dash="dash", annotation_text="90%")
    fig.update_layout(title=t(viz.label), yaxis_title="Cumulative Variance", template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_cumulative_variance")


def _viz_projection_2d(model, X, y, task, feature_names, viz):
    X_proj = model.transform(X) if hasattr(model, 'transform') else model.fit_transform(X)
    fig = go.Figure()
    if y is not None:
        for cls in np.unique(y):
            mask = y == cls
            fig.add_trace(go.Scatter(x=X_proj[mask, 0], y=X_proj[mask, 1] if X_proj.shape[1] > 1 else np.zeros(mask.sum()),
                                      mode='markers', name=f'Class {cls}', marker=dict(size=8)))
    else:
        fig.add_trace(go.Scatter(x=X_proj[:, 0], y=X_proj[:, 1] if X_proj.shape[1] > 1 else np.zeros(len(X_proj)),
                                  mode='markers', marker=dict(size=8, color='#3498db')))
    fig.update_layout(title=t(viz.label), xaxis_title="PC1", yaxis_title="PC2", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_projection_2d")


def _viz_loadings_heatmap(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'components_'):
        return
    comp = model.components_
    n_feat = min(comp.shape[1], len(feature_names))
    fig = px.imshow(comp[:, :n_feat], text_auto=".2f", x=feature_names[:n_feat],
                    y=[f"PC{i+1}" for i in range(comp.shape[0])], color_continuous_scale='RdBu_r')
    fig.update_layout(title=t(viz.label), height=300)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_loadings_heatmap")


def _viz_class_distribution(model, X, y, task, feature_names, viz):
    if y is None:
        return
    unique, counts = np.unique(y, return_counts=True)
    fig = go.Figure(go.Bar(x=[str(u) for u in unique], y=counts, marker_color='#3498db'))
    fig.update_layout(title=t(viz.label), xaxis_title=t("viz.class"), yaxis_title="Count", template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_class_distribution")


def _viz_roc_curve(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'predict_proba') or len(np.unique(y)) != 2:
        st.caption("ROC curve available only for binary classification with predict_proba.")
        return
    from sklearn.metrics import roc_curve, auc
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='#e74c3c', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
    fig.update_layout(title=t(viz.label), xaxis_title="FPR", yaxis_title="TPR", template='plotly_white', height=350)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_roc_curve")


def _viz_probability_distribution(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'predict_proba'):
        return
    
    y_proba = model.predict_proba(X)[:, 1]
    fig = go.Figure()
    
    if y is not None:
        for cls in np.unique(y):
            mask = y == cls
            fig.add_trace(go.Histogram(x=y_proba[mask], name=f'Klasa {cls}', opacity=0.7, nbinsx=20))
    else:
        fig.add_trace(go.Histogram(x=y_proba, name='Prawdopodobieństwa', opacity=0.7, nbinsx=20))
        
    fig.update_layout(title=t(viz.label), xaxis_title=f"{t('viz.predicted')} P(y=1)", yaxis_title="Count", 
                      barmode='overlay', template='plotly_white', height=350)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_prob_dist")


def _viz_precision_recall_curve(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'predict_proba') or len(np.unique(y)) != 2:
        return
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_proba)
    ap = average_precision_score(y, y_proba)
    
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AP={ap:.3f})', line=dict(color='#8e44ad', width=2)))
    fig.update_layout(title=t(viz.label), xaxis_title=f"Recall", yaxis_title=f"Precision", template='plotly_white', height=350)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="viz_pr_curve")


def _viz_support_vectors(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'support_vectors_'):
        return
    sv = model.support_vectors_
    st.caption(f"Support vectors: {len(sv)} points highlighted on the main chart.")


def _viz_data_table(model, X, y, task, feature_names, viz):
    """Renders a simple table of the input dataset (X and y)."""
    df = pd.DataFrame(X, columns=feature_names)
    if y is not None:
        target_name = st.session_state.get("last_run_target", "Target")
        df[target_name] = y
    
    st.markdown(f"**📂 {t(viz.label)}**")
    st.dataframe(df, use_container_width=True)


def render_empty_results_panel(config: PluginConfig, X: Optional[np.ndarray], y: Optional[np.ndarray], feature_names: List[str]):
    """Renders the skeleton layout of the results panel before the model runs."""
    meta = config.metadata
    visible_vizs = [v for v in config.visualizations if v.show]
    main_vizs = [v for v in visible_vizs if v.position == "main"]
    side_vizs = [v for v in visible_vizs if v.position == "side"]
    bottom_vizs = [v for v in visible_vizs if v.position == "bottom"]

    st.markdown(f"#### 🖼️ {t('viz.visualizations_header')}")
    
    # If X is provided, show the actual data preview ONCE at the top
    if X is not None:
        outliers = st.session_state.get("current_outliers")
        has_outliers = outliers is not None and len(outliers) == len(X)
        
        with st.container(border=True):
            if meta.task == "regression":
                if X.shape[1] == 1:
                    preview_fig = go.Figure()
                    if has_outliers:
                        preview_fig.add_trace(go.Scatter(x=X[~outliers, 0], y=y[~outliers], mode='markers', name='Dane (Normalne)', marker=dict(size=8, color='#3498db', opacity=0.7)))
                        preview_fig.add_trace(go.Scatter(x=X[outliers, 0], y=y[outliers], mode='markers', name='Elementy Odstające', marker=dict(size=10, color='#e74c3c', opacity=0.9, symbol='x')))
                    else:
                        preview_fig.add_trace(go.Scatter(x=X[:, 0], y=y, mode='markers', name='Dane', marker=dict(size=8, color='#3498db', opacity=0.7)))
                    
                    title = "Podgląd danych wejściowych"
                    xtit = feature_names[0] if feature_names else "Cecha 1 (X)"
                else: 
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=1)
                    X_pca = pca.fit_transform(X)
                    preview_fig = go.Figure()
                    if has_outliers:
                        preview_fig.add_trace(go.Scatter(x=X_pca[~outliers, 0], y=y[~outliers], mode='markers', name='Dane (Normalne)', marker=dict(size=8, color='#3498db', opacity=0.7)))
                        preview_fig.add_trace(go.Scatter(x=X_pca[outliers, 0], y=y[outliers], mode='markers', name='Elementy Odstające', marker=dict(size=10, color='#e74c3c', opacity=0.9, symbol='x')))
                    else:
                        preview_fig.add_trace(go.Scatter(x=X_pca[:, 0], y=y, mode='markers', name='Dane', marker=dict(size=8, color='#3498db', opacity=0.7)))
                        
                    title = "Podgląd: Cel vs Główny Składnik PCA (Redukcja wymiaru do X)"
                    xtit = "Główny Składnik 1 (PCA)"
                
                preview_fig.update_layout(title=title, xaxis_title=xtit, yaxis_title="Cel (y)", template='plotly_white')
                st.plotly_chart(preview_fig, use_container_width=True, config={'displayModeBar': False}, key="empty_data_preview")
                
            elif meta.task == "classification":
                if X.shape[1] == 1:
                    preview_fig = go.Figure()
                    for cls in np.unique(y):
                        mask = y == cls
                        preview_fig.add_trace(go.Histogram(x=X[mask, 0], name=f'Klasa {cls}', opacity=0.7))
                    preview_fig.update_layout(title="Podgląd: Rozkład klas dla pojedynczej cechy", 
                                            xaxis_title=feature_names[0] if feature_names else "Cecha", 
                                            barmode='overlay', template='plotly_white')
                elif X.shape[1] == 2:
                    preview_fig = go.Figure()
                    for cls in np.unique(y):
                        mask = y == cls
                        preview_fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=f'Klasa {cls}', marker=dict(size=8)))
                    preview_fig.update_layout(title="Podgląd wczytanych danych w 2D", 
                                            xaxis_title=feature_names[0] if feature_names else "Cecha 1",
                                            yaxis_title=feature_names[1] if feature_names else "Cecha 2",
                                            template='plotly_white')
                else:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X)
                    preview_fig = go.Figure()
                    for cls in np.unique(y):
                        mask = y == cls
                        preview_fig.add_trace(go.Scatter(x=X_pca[mask, 0], y=X_pca[mask, 1], mode='markers', name=f'Klasa {cls}', marker=dict(size=8)))
                    preview_fig.update_layout(title="Podgląd danych wielowymiarowych (Rzut PCA w 2D)",
                                            xaxis_title="Zmienna Latentna 1 (PCA)",
                                            yaxis_title="Zmienna Latentna 2 (PCA)", template='plotly_white')
                    
                st.plotly_chart(preview_fig, use_container_width=True, config={'displayModeBar': False}, key="empty_data_preview")

    # Create the generic grey placeholder for the actual model visuals
    placeholder_fig = go.Figure()
    if X is None:
        placeholder_fig.update_layout(
            template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(text=t("viz.load_data_hint"), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#bbb"))]
        )
    else:
        placeholder_fig.update_layout(
            template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(text=t("viz.run_model_hint"), xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#bbb"))]
        )

    # --- MAIN VIZS ---
    if main_vizs and X is None:
        for i, viz in enumerate(main_vizs):
            with st.container(border=True):
                f = go.Figure(placeholder_fig)
                f.update_layout(title=f"{t(viz.label)} ({t('viz.waiting')})")
                st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False}, key=f"empty_main_{i}")

    # --- EQUATION ---
    eq_vizs = [v for v in visible_vizs if v.name == "equation" and v.position == "top"]
    if eq_vizs:
        _inject_equation_styles()
        # Header Bar (Always visible)
        st.markdown(f'<div class="eq-header-bar">📐 {t("viz.learned_eq").upper()}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            if meta.task == "regression":
                st.latex(r"y = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n + \epsilon")
            elif meta.task == "classification":
                st.latex(r"P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots)}}")
            else:
                st.markdown(f'<div style="padding:10px 0; color:#94a3b8; text-align:center;">{t("viz.waiting")}</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="eq-footer-row">
                    <span style="color:#94a3b8;"><b>{t("viz.coef_label")}</b>: ?</span>
                    <span style="color:#94a3b8;"><b>{t("viz.intercept_label")}</b>: ?</span>
                </div>
            """, unsafe_allow_html=True)

    # --- METRICS ---
    visible_metrics = [m for m in config.metrics if m.show]
    if visible_metrics:
        with st.container(border=True):
            st.markdown(f"**📈 {t('viz.quality_metrics')}**")
            inner_cols = st.columns(len(visible_metrics))
            for i, m in enumerate(visible_metrics):
                inner_cols[i].metric(label=t(m.label), value="0.0000e+00" if m.format != "percent" else "0%")
                if m.hint:
                    inner_cols[i].caption(t(m.hint)[:100])

    # --- BOTTOM VISUALIZATIONS ---
    # Stacked vertically
    if bottom_vizs:
        for i, viz in enumerate(bottom_vizs):
            with st.container(border=True):
                if viz.name == "data_table":
                    if X is not None:
                        # Render the actual table if data is already loaded
                        df = pd.DataFrame(X, columns=feature_names)
                        if y is not None:
                            target_name = st.session_state.get("current_target", "Target")
                            df[target_name] = y
                        st.markdown(f"**📂 {t(viz.label)}**")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown(f"**📂 {t(viz.label)} ({t('viz.waiting')})**")
                        st.dataframe(pd.DataFrame({"...": ["..."]}), use_container_width=True)
                else:
                    f = go.Figure(placeholder_fig)
                    f.update_layout(title=f"{t(viz.label)} ({t('viz.waiting')})")
                    st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False}, key=f"empty_bottom_{i}")


def render_side_visualizations(config: PluginConfig, model, X, y, task, feature_names):
    """Renders side-positioned visualizations (e.g. for the left column)."""
    side_vizs = [v for v in config.visualizations if v.show and v.position == "side"]
    for viz in side_vizs:
        _render_viz(viz, model, X, y, task, feature_names)


def render_side_visualizations_skeleton(config: PluginConfig, X_available: bool):
    """Renders side-positioned visualization skeletons."""
    side_vizs = [v for v in config.visualizations if v.show and v.position == "side"]
    if not side_vizs:
        return
        
    placeholder_fig = go.Figure()
    text = "Wczytaj dane podgląd..." if not X_available else "Kliknij 'Run Model'..."
    placeholder_fig.update_layout(
        template='plotly_white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[dict(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#bbb"))]
    )

    for i, viz in enumerate(side_vizs):
        with st.container(border=True):
            f = go.Figure(placeholder_fig)
            f.update_layout(title=f"{viz.label} ({t('viz.waiting')})")
            st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False}, key=f"empty_side_skeleton_{i}")


def render_model_attributes_card(config: PluginConfig, model, feature_names: List[str]):
    """Renders the Model Attributes card for the left column."""
    outputs = get_model_outputs(model, feature_names, config)
    if outputs:
        with st.container(border=True):
            st.markdown(f"**📊 {t('viz.attributes')}**")
            for attr_name, odata in outputs.items():
                _render_output(attr_name, odata, feature_names)


def render_model_attributes_skeleton(config: PluginConfig):
    """Renders the Model Attributes skeleton for the left column."""
    visible_outputs = [o for o in config.outputs if o.show]
    if visible_outputs:
        with st.container(border=True):
            st.markdown(f"**📊 {t('viz.attributes')}**")
            for o in visible_outputs:
                st.markdown(f"**📋 {o.label}**")
                st.write("—")
                st.markdown("<hr style='margin:12px 0; border:none; border-top:1px solid #f0f0f0'>", unsafe_allow_html=True)

