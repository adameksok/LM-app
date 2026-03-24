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


def render_data_preview(task: str, X: np.ndarray, y: Optional[np.ndarray], feature_names: List[str]):
    """Renders a scatter plot of the raw data before the model is run."""
    st.markdown("#### 🖼️ Podgląd wczytanych danych")
    fig = go.Figure()

    if task == "regression":
        X_flat = X.reshape(-1, 1) if X.ndim == 1 else X
        fig.add_trace(go.Scatter(
            x=X_flat[:, 0], y=y, mode='markers', name='Dane',
            marker=dict(size=8, color='#3498db', opacity=0.7)
        ))
        fig.update_layout(
            title="Wykres punktowy",
            xaxis_title=feature_names[0] if feature_names else "X",
            yaxis_title="y",
            template='plotly_white'
        )
    elif task == "classification":
        if X.shape[1] >= 2:
            for cls in np.unique(y):
                mask = y == cls
                fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=f'Klasa {cls}',
                                         marker=dict(size=8, line=dict(width=1, color='white'))))
            fig.update_layout(title="Rozkład klas", xaxis_title=feature_names[0] if feature_names else "X₁", 
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

    st.plotly_chart(fig, use_container_width=True)


def render_results_panel(config: PluginConfig, model, X, y, params, feature_names):
    """Main entry — renders the full results panel after model.fit()."""

    task = config.metadata.task

    # Apply params and fit model
    try:
        model.set_params(**params)
    except Exception:
        pass

    if task == "dimensionality_reduction":
        model.fit(X)
    elif task == "clustering":
        model.fit(X)
    else:
        model.fit(X, y)
        
    # --- MAIN VISUALIZATIONS ---
    visible_vizs = [v for v in config.visualizations if v.show]
    main_vizs = [v for v in visible_vizs if v.position == "main"]
    side_vizs = [v for v in visible_vizs if v.position == "side"]
    bottom_vizs = [v for v in visible_vizs if v.position == "bottom"]

    for viz in main_vizs:
        _render_viz(viz, model, X, y, task, feature_names)

    if side_vizs:
        cols = st.columns(len(side_vizs))
        for col, viz in zip(cols, side_vizs):
            with col:
                _render_viz(viz, model, X, y, task, feature_names)

    # --- EQUATION ---
    eq_vizs = [v for v in visible_vizs if v.name == "equation" and v.position == "top"]
    if eq_vizs:
        equation = build_equation(model, feature_names, task)
        if equation:
            with st.container(border=True):
                st.markdown("**📐 Wyuczone równanie**")
                st.latex(equation.replace("·", r" \cdot ").replace("ŷ", r"\hat{y}"))

    # --- METRICS ---
    metrics = calculate_metrics(model, X, y, task, config)
    if metrics:
        with st.container(border=True):
            st.markdown("**📈 Metryki jakości**")
            cols = st.columns(len(metrics))
            for col, (mid, mdata) in zip(cols, metrics.items()):
                value = mdata['value']
                label = mdata['label']
                fmt = mdata['format']
                good = mdata.get('good_value')

                if fmt == "percent":
                    display_val = f"{value:.1%}"
                elif fmt == "integer":
                    display_val = str(int(value))
                else:
                    display_val = f"{value:.4f}"

                delta = None
                if good is not None and isinstance(value, (int, float)):
                    delta = f"{'✅ good' if value >= good else '⚠️ low'}"

                col.metric(label=label, value=display_val, delta=delta)

                if mdata.get('hint'):
                    col.caption(mdata['hint'][:100])

    # --- MODEL ATTRIBUTES / OUTPUTS ---
    outputs = get_model_outputs(model, feature_names, config)
    if outputs:
        st.markdown("#### 📊 Atrybuty modelu")
        for attr_name, odata in outputs.items():
            _render_output(attr_name, odata, feature_names)

    # --- BOTTOM VISUALIZATIONS ---
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

    with st.expander(f"📋 {label}", expanded=False):
        if fmt == "text":
            if isinstance(value, np.ndarray):
                st.code(str(value))
            else:
                st.write(f"**{value}**")

        elif fmt == "bar_chart":
            if isinstance(value, np.ndarray) and value.ndim == 1:
                names = feature_names[:len(value)] if len(feature_names) >= len(value) else [f"f{i}" for i in range(len(value))]
                fig = go.Figure(go.Bar(x=names, y=value, marker_color=['#e74c3c' if v < 0 else '#3498db' for v in value]))
                fig.update_layout(title=label, template='plotly_white', height=300)
                st.plotly_chart(fig, use_container_width=True)
            elif isinstance(value, np.ndarray) and value.ndim == 2:
                fig = go.Figure(go.Bar(x=[f"f{i}" for i in range(value.shape[1])], y=value[0]))
                fig.update_layout(title=label, template='plotly_white', height=300)
                st.plotly_chart(fig, use_container_width=True)
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
                fig.update_layout(title=label, height=300)
                st.plotly_chart(fig, use_container_width=True)

        elif fmt == "scatter_overlay":
            st.caption(f"{len(value)} items")
            st.code(str(value[:5]) + (" ..." if len(value) > 5 else ""))

        else:
            st.write(f"**{value}**")

        if hint:
            st.info(f"💡 {hint}")


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
        "support_vectors_overlay": _viz_support_vectors,
    }

    renderer = viz_map.get(viz.name)
    if renderer:
        try:
            renderer(model, X, y, task, feature_names, viz)
        except Exception as e:
            st.warning(f"⚠️ {viz.label}: {e}")


def _viz_decision_boundary(model, X, y, task, feature_names, viz):
    if X.shape[1] < 2:
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
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=f'Class {cls}',
                                  marker=dict(size=8, line=dict(width=1, color='white'))))
    fig.update_layout(title=viz.label, xaxis_title="X₁", yaxis_title="X₂", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def _viz_regression_fit(model, X, y, task, feature_names, viz):
    """Scatter plot of data points + fitted regression line."""
    X_flat = X.reshape(-1, 1) if X.ndim == 1 else X
    y_pred = model.predict(X_flat)

    fig = go.Figure()

    # Data points
    fig.add_trace(go.Scatter(
        x=X_flat[:, 0], y=y, mode='markers', name='Dane',
        marker=dict(size=8, color='#3498db', opacity=0.7)
    ))

    # Fitted line (sorted for smooth line)
    x_sorted = np.sort(X_flat[:, 0])
    y_line = model.predict(x_sorted.reshape(-1, 1))
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_line, mode='lines', name='Dopasowanie',
        line=dict(color='#e74c3c', width=3)
    ))

    fig.update_layout(
        title=viz.label,
        xaxis_title=feature_names[0] if feature_names else "X",
        yaxis_title="y",
        hovermode='closest',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)


def _viz_confusion_matrix(model, X, y, task, feature_names, viz):
    y_pred = model.predict(X)
    cm = sk_confusion_matrix(y, y_pred)
    labels = sorted(np.unique(y))
    fig = px.imshow(cm, text_auto=True, x=[str(l) for l in labels], y=[str(l) for l in labels],
                    color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(title=viz.label, height=350)
    st.plotly_chart(fig, use_container_width=True)


def _viz_coefficients_bar(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'coef_'):
        return
    coef = model.coef_.flatten()[:len(feature_names)]
    fig = go.Figure(go.Bar(x=feature_names[:len(coef)], y=coef,
                           marker_color=['#e74c3c' if v < 0 else '#3498db' for v in coef]))
    fig.update_layout(title=viz.label, template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True)


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
    fig.update_layout(title=viz.label, xaxis_title="Predicted", yaxis_title="Residual", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


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
    fig.update_layout(title=viz.label, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def _viz_variance_bar(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'explained_variance_ratio_'):
        return
    evr = model.explained_variance_ratio_
    fig = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(evr))], y=evr, marker_color='#3498db'))
    fig.update_layout(title=viz.label, yaxis_title="Variance Ratio", template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True)


def _viz_cumulative_variance(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'explained_variance_ratio_'):
        return
    evr = model.explained_variance_ratio_
    cumul = np.cumsum(evr)
    fig = go.Figure(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumul))], y=cumul, mode='lines+markers',
                                marker=dict(size=8, color='#e74c3c')))
    fig.add_hline(y=0.9, line_dash="dash", annotation_text="90%")
    fig.update_layout(title=viz.label, yaxis_title="Cumulative Variance", template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True)


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
    fig.update_layout(title=viz.label, xaxis_title="PC1", yaxis_title="PC2", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def _viz_loadings_heatmap(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'components_'):
        return
    comp = model.components_
    n_feat = min(comp.shape[1], len(feature_names))
    fig = px.imshow(comp[:, :n_feat], text_auto=".2f", x=feature_names[:n_feat],
                    y=[f"PC{i+1}" for i in range(comp.shape[0])], color_continuous_scale='RdBu_r')
    fig.update_layout(title=viz.label, height=300)
    st.plotly_chart(fig, use_container_width=True)


def _viz_class_distribution(model, X, y, task, feature_names, viz):
    if y is None:
        return
    unique, counts = np.unique(y, return_counts=True)
    fig = go.Figure(go.Bar(x=[str(u) for u in unique], y=counts, marker_color='#3498db'))
    fig.update_layout(title=viz.label, xaxis_title="Class", yaxis_title="Count", template='plotly_white', height=300)
    st.plotly_chart(fig, use_container_width=True)


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
    fig.update_layout(title=viz.label, xaxis_title="FPR", yaxis_title="TPR", template='plotly_white', height=350)
    st.plotly_chart(fig, use_container_width=True)


def _viz_support_vectors(model, X, y, task, feature_names, viz):
    if not hasattr(model, 'support_vectors_'):
        return
    sv = model.support_vectors_
    st.caption(f"Support vectors: {len(sv)} points highlighted on the main chart.")


def render_empty_results_panel(config: PluginConfig, X: Optional[np.ndarray], y: Optional[np.ndarray], feature_names: List[str]):
    """Renders the skeleton layout of the results panel before the model runs."""
    meta = config.metadata
    visible_vizs = [v for v in config.visualizations if v.show]
    main_vizs = [v for v in visible_vizs if v.position == "main"]
    side_vizs = [v for v in visible_vizs if v.position == "side"]
    bottom_vizs = [v for v in visible_vizs if v.position == "bottom"]

    st.markdown("#### 🖼️ Wizualizacje")
    
    # If X is provided, show the actual data preview. Otherwise, show empty grid.
    if X is not None:
        if meta.task == "regression":
            X_flat = X.reshape(-1, 1) if X.ndim == 1 else X
            fig = go.Figure(go.Scatter(
                x=X_flat[:, 0], y=y, mode='markers', name='Dane',
                marker=dict(size=8, color='#3498db', opacity=0.7)
            ))
            fig.update_layout(title="Podgląd danych", template='plotly_white')
        elif meta.task == "classification" and X.shape[1] >= 2:
            fig = go.Figure()
            for cls in np.unique(y):
                mask = y == cls
                fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name=f'Klasa {cls}', marker=dict(size=8)))
            fig.update_layout(title="Rozkład klas", template='plotly_white')
        else:
            fig = go.Figure()
            fig.update_layout(title="Podgląd danych nedostępny", template='plotly_white')
    else:
        fig = go.Figure()
        fig.update_layout(
            title="Brak danych do wyświetlenia", template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(text="Wczytaj dane z panelu po lewej stronie", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#bbb"))]
        )

    if main_vizs:
        st.plotly_chart(fig, use_container_width=True)

    if side_vizs:
        cols = st.columns(len(side_vizs))
        for col in cols:
            with col:
                st.plotly_chart(fig, use_container_width=True)

    # --- EQUATION ---
    eq_vizs = [v for v in visible_vizs if v.name == "equation" and v.position == "top"]
    if eq_vizs:
        with st.container(border=True):
            st.markdown("**📐 Wyuczone równanie**")
            if meta.task == "regression":
                st.latex(r"y = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n")
            elif meta.task == "classification":
                st.latex(r"P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots)}}")
            else:
                st.latex(r"f(X) = ?")

    # --- METRICS ---
    visible_metrics = [m for m in config.metrics if m.show]
    if visible_metrics:
        with st.container(border=True):
            st.markdown("**📈 Metryki jakości**")
            cols = st.columns(len(visible_metrics))
            for col, m in zip(cols, visible_metrics):
                col.metric(label=m.label, value="0.00" if m.format != "percent" else "0%")
                if m.hint:
                    col.caption(m.hint[:100])

    # --- MODEL ATTRIBUTES / OUTPUTS ---
    visible_outputs = [o for o in config.outputs if o.show]
    if visible_outputs:
        st.markdown("#### 📊 Atrybuty modelu")
        for o in visible_outputs:
            with st.expander(f"📋 {o.label}", expanded=False):
                st.write("**—**")

    # --- BOTTOM VISUALIZATIONS ---
    if bottom_vizs:
        for viz in bottom_vizs:
            st.plotly_chart(fig, use_container_width=True)
