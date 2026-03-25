# app.py — Entry point
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Edu-ML Sandbox",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

from components.dashboard import render_dashboard
from components.visualization import render_results_panel, render_empty_results_panel
from core.plugin_engine import get_plugin_engine
from core.session_state import init_session_state
from core.plugin_interface import ParameterConfig
from typing import List, Dict, Any

# ── Custom CSS ──
APP_CSS = """
<style>
/* ── Sidebar branding ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8ECF0;
}
.sidebar-brand {
    padding: 8px 0 4px 0;
}
.sidebar-brand h2 {
    margin: 0; font-size: 18px; font-weight: 800; color: #1A1A2E;
}
.sidebar-brand .sub {
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 1.5px; color: #999; margin-top: 2px;
}
.sidebar-nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 8px; color: #555;
    font-size: 14px; font-weight: 500; text-decoration: none;
    margin-bottom: 2px; cursor: pointer;
}
.sidebar-nav-item.active {
    background: #EBF5FF; color: #1976D2; font-weight: 600;
}
.sidebar-nav-item:hover { background: #F5F7FA; }

/* ── Param chip ── */
.param-chip {
    background: #F5F7FA; border: 1px solid #E8ECF0;
    border-radius: 8px; padding: 10px 16px; text-align: center;
}
.param-chip .lbl {
    font-size: 9px; text-transform: uppercase;
    letter-spacing: 1px; color: #999; margin-bottom: 2px;
}
.param-chip .val {
    font-size: 18px; font-weight: 700; color: #1A1A2E;
}

/* ── Data dropzone ── */
.data-dropzone {
    border: 2px dashed #D0D5DD; border-radius: 12px;
    padding: 36px 20px; text-align: center; background: #FAFBFC;
    transition: border-color 0.2s;
}
.data-dropzone:hover { border-color: #1976D2; }
.data-dz-icon { font-size: 32px; color: #888; margin-bottom: 10px; }
.data-dz-title { font-size: 15px; font-weight: 600; color: #1A1A2E; margin-bottom: 4px; }
.data-dz-sub { font-size: 12px; color: #888; }
.data-formats { display: flex; gap: 8px; margin-top: 14px; }
.data-fmt {
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    padding: 3px 10px; border-radius: 4px; background: #F0F2F5; color: #666;
}

/* ── Bottom bar ── */
.bottom-info {
    font-size: 12px; color: #888; display: flex; align-items: center; gap: 6px;
}
.bottom-dot {
    width: 7px; height: 7px; border-radius: 50%; display: inline-block;
}

/* ── Settings table ── */
.settings-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.settings-table th {
    text-align: left; padding: 8px 12px; background: #F5F7FA;
    border-bottom: 2px solid #E8ECF0; color: #555; font-weight: 600;
}
.settings-table td {
    padding: 7px 12px; border-bottom: 1px solid #F0F2F5; color: #333;
}
.settings-table tr:hover { background: #FAFBFC; }
.tag-badge {
    font-size: 10px; font-weight: 700; padding: 2px 8px;
    border-radius: 4px; display: inline-block;
}

/* ── White Widget Backgrounds ── */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stExpander"] {
    background-color: #ffffff;
}
</style>
"""


def main():
    init_session_state()

    if st.session_state.current_view == "dashboard":
        render_dashboard()
    elif st.session_state.current_view == "experiment":
        render_experiment_view()
    elif st.session_state.current_view == "saved_model":
        from components.saved_model_view import render_saved_model_view
        render_saved_model_view()


def _render_sidebar_nav(meta):
    """Sidebar: branding + navigation + actions."""

    st.markdown("""
    <div class="sidebar-brand">
        <h2>ML Insights</h2>
        <div class="sub">Insight Architect</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    if st.button("🏠  Models", use_container_width=True):
        st.session_state.current_view = "dashboard"
        st.cache_resource.clear()
        st.rerun()

    st.button("📊  Data", use_container_width=True, disabled=True)
    st.button("📈  Analysis", use_container_width=True, disabled=True)

    # Spacer
    st.markdown("<br>" * 8, unsafe_allow_html=True)

    # New experiment
    if st.button("＋ New Experiment", type="primary", use_container_width=True):
        st.session_state.current_view = "dashboard"
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("📖 Documentation")
    st.caption("💬 Support")


def _render_param_controls(parameters: List[ParameterConfig]) -> Dict[str, Any]:
    """Renders parameter controls inline (not in sidebar). Returns params dict."""
    params = {}

    if not parameters:
        st.caption("Brak konfigurowalnych parametrów.")
        return params

    for param in parameters:
        if not param.show:
            continue

        if param.type == "int":
            params[param.name] = st.slider(
                label=param.label,
                min_value=int(param.min_val),
                max_value=int(param.max_val),
                value=int(param.default),
                step=int(param.step) if param.step else 1,
                help=param.hint
            )
        elif param.type == "float":
            params[param.name] = st.slider(
                label=param.label,
                min_value=float(param.min_val),
                max_value=float(param.max_val),
                value=float(param.default),
                step=float(param.step) if param.step else 0.01,
                help=param.hint
            )
        elif param.type == "bool":
            params[param.name] = st.checkbox(
                label=param.label,
                value=bool(param.default),
                help=param.hint
            )
        elif param.type == "select":
            options = param.options or []
            default_idx = options.index(param.default) if param.default in options else 0
            
            def format_func(x, p=param):
                opt_labels = getattr(p, 'option_labels', None)
                if opt_labels and x in opt_labels:
                    return opt_labels[x]
                return str(x)
                
            params[param.name] = st.selectbox(
                label=param.label,
                options=options,
                index=default_idx,
                format_func=format_func,
                help=param.hint
            )

    return params


def _render_settings_tab(config):
    """Settings tab: display plugin configuration from .py file."""
    from pathlib import Path

    meta = config.metadata

    st.markdown("#### 📋 Plugin Configuration")

    # Source file
    st.markdown(f"**Source file:** `{Path(config.source_file).name}`")

    st.divider()

    # Metadata
    st.markdown("##### 📌 Metadata")
    meta_data = {
        "Model Class": meta.model_class,
        "Task": meta.task,
        "Name": meta.name,
        "Description": meta.description[:120] + "..." if len(meta.description) > 120 else meta.description,
        "Icon": meta.icon or "—",
    }
    meta_html = '<table class="settings-table"><tr><th>Property</th><th>Value</th></tr>'
    for k, v in meta_data.items():
        meta_html += f'<tr><td>{k}</td><td>{v}</td></tr>'
    meta_html += '</table>'
    st.markdown(meta_html, unsafe_allow_html=True)

    st.divider()

    # Parameters
    st.markdown(f"##### ⚙️ Parameters ({len(config.parameters)})")
    if config.parameters:
        param_html = '<table class="settings-table"><tr><th>Name</th><th>Label</th><th>Type</th><th>Default</th><th>Show</th></tr>'
        for p in config.parameters:
            show_badge = '<span class="tag-badge" style="background:#E8F5E9;color:#2E7D32;">Show</span>' if p.show else '<span class="tag-badge" style="background:#FFF3E0;color:#E65100;">Hidden</span>'
            param_html += f'<tr><td><code>{p.name}</code></td><td>{p.label}</td><td>{p.type}</td><td>{p.default}</td><td>{show_badge}</td></tr>'
        param_html += '</table>'
        st.markdown(param_html, unsafe_allow_html=True)

    st.divider()

    # Outputs
    st.markdown(f"##### 📊 Outputs ({len(config.outputs)})")
    if config.outputs:
        out_html = '<table class="settings-table"><tr><th>Attribute</th><th>Label</th><th>Format</th><th>Show</th></tr>'
        for o in config.outputs:
            show_badge = '<span class="tag-badge" style="background:#E8F5E9;color:#2E7D32;">Show</span>' if o.show else '<span class="tag-badge" style="background:#FFF3E0;color:#E65100;">Hidden</span>'
            out_html += f'<tr><td><code>{o.name}</code></td><td>{o.label}</td><td>{o.format}</td><td>{show_badge}</td></tr>'
        out_html += '</table>'
        st.markdown(out_html, unsafe_allow_html=True)
    else:
        st.caption("Defaults per @task")

    st.divider()

    # Metrics
    st.markdown(f"##### 📈 Metrics ({len(config.metrics)})")
    if config.metrics:
        met_html = '<table class="settings-table"><tr><th>ID</th><th>Label</th><th>Format</th><th>Good ≥</th><th>Show</th></tr>'
        for m in config.metrics:
            show_badge = '<span class="tag-badge" style="background:#E8F5E9;color:#2E7D32;">Show</span>' if m.show else '<span class="tag-badge" style="background:#FFF3E0;color:#E65100;">Hidden</span>'
            gv = f"{m.good_value}" if m.good_value is not None else "—"
            met_html += f'<tr><td><code>{m.name}</code></td><td>{m.label}</td><td>{m.format}</td><td>{gv}</td><td>{show_badge}</td></tr>'
        met_html += '</table>'
        st.markdown(met_html, unsafe_allow_html=True)

    st.divider()

    # Visualizations
    st.markdown(f"##### 🖼️ Visualizations ({len(config.visualizations)})")
    if config.visualizations:
        viz_html = '<table class="settings-table"><tr><th>ID</th><th>Label</th><th>Position</th><th>Show</th></tr>'
        for v in config.visualizations:
            show_badge = '<span class="tag-badge" style="background:#E8F5E9;color:#2E7D32;">Show</span>' if v.show else '<span class="tag-badge" style="background:#FFF3E0;color:#E65100;">Hidden</span>'
            viz_html += f'<tr><td><code>{v.name}</code></td><td>{v.label}</td><td>{v.position}</td><td>{show_badge}</td></tr>'
        viz_html += '</table>'
        st.markdown(viz_html, unsafe_allow_html=True)

    # Plugin source viewer
    st.divider()
    st.markdown("##### 📝 Plugin Source")
    with st.expander("View source code", expanded=False):
        try:
            source = Path(config.source_file).read_text(encoding="utf-8")
            st.code(source, language="python")
        except Exception:
            st.warning("Cannot read source file.")


def render_experiment_view():
    st.markdown(APP_CSS, unsafe_allow_html=True)

    engine = get_plugin_engine()
    config = engine.get_plugin(st.session_state.current_model_id)

    # Flush session state to prevent contamination when switching models
    if st.session_state.get("_last_experiment_model_id") != st.session_state.current_model_id:
        st.session_state["_last_experiment_model_id"] = st.session_state.current_model_id
        st.session_state["model_ran"] = False
        keys_to_clear = ["last_run_X", "last_run_y", "last_run_params", "last_run_features", "last_run_target",
                         "current_X", "current_y", "current_features", "current_target"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]

    if not config:
        st.error("❌ Plugin not found!")
        if st.button("← Back to Dashboard"):
            st.session_state.current_view = "dashboard"
            st.rerun()
        return

    meta = config.metadata

    # === SIDEBAR ===
    with st.sidebar:
        _render_sidebar_nav(meta)

    # === TOP TABS ===
    tab_setup, tab_history, tab_settings = st.tabs(["⚡ Model Setup", "📜 History", "⚙️ Settings"])

    # ================================================================
    # TAB: Model Setup — Option B: Control Panel (left) + Results (right)
    # ================================================================
    with tab_setup:
        # Read sklearn defaults for param chips
        sklearn_params = {}
        if hasattr(config.model_instance, "get_params"):
            sklearn_params = config.model_instance.get_params()
        visible_params = [p for p in config.parameters if p.show]
        last_params = st.session_state.get("last_run_params", {})

        # ── TWO-COLUMN LAYOUT: 1/3 controls, 2/3 results ──
        col_ctrl, col_results = st.columns([1, 2], gap="large")

        # Prep variables for both columns
        run_X = st.session_state.get("last_run_X")
        run_y = st.session_state.get("last_run_y")
        run_params = st.session_state.get("last_run_params", {})
        run_features = st.session_state.get("last_run_features", [])

        from components.visualization import (
            fit_model_instance, render_results_panel, render_empty_results_panel,
            render_model_attributes_card, render_model_attributes_skeleton,
            render_side_visualizations, render_side_visualizations_skeleton
        )
        
        # Ensure model is fitted before rendering either column for consistency
        if st.session_state.get("model_ran", False) and run_X is not None:
             fit_model_instance(config, config.model_instance, run_X, run_y, run_params)

        # ==============================================================
        # LEFT COLUMN — Control Panel
        # ==============================================================
        with col_ctrl:
            # ── Architecture (compact) ──
            with st.container(border=True):
                hcol1, hcol2 = st.columns([3, 1])
                with hcol1:
                    st.markdown("**⚙️ Architecture**")
                with hcol2:
                    st.markdown(
                        '<span style="font-size:11px;font-weight:700;letter-spacing:0.5px;'
                        'padding:4px 12px;border-radius:999px;background:#E8F5E9;color:#2E7D32;">'
                        'CONFIGURED</span>',
                        unsafe_allow_html=True
                    )

                task_icons = {"classification": "🎯", "regression": "📈", "clustering": "🔄", "dimensionality_reduction": "📊"}
                st.markdown(f"**{meta.model_class}** · {task_icons.get(meta.task, '📊')} {meta.task}")

                # Param chips (compact row)
                preview_params = visible_params[:3]
                if preview_params:
                    chip_cols = st.columns(len(preview_params))
                    for i, p in enumerate(preview_params):
                        with chip_cols[i]:
                            label = p.name.upper().replace("_", " ")
                            if len(label) > 10:
                                label = label[:10]
                            val = last_params.get(p.name, sklearn_params.get(p.name, p.default))
                            st.metric(label=label, value=str(val))

            # Placeholder for Run Model button (reserving visual space)
            run_action_placeholder = st.empty()

            # ── Adjust Parameters ──
            with st.expander("☰ Adjust Parameters", expanded=False):
                params = _render_param_controls(config.parameters)

            # ── Load Data ──
            with st.container(border=True):
                st.markdown("**📁 Load Data**")
                raw_df, source_type = _render_data_card(meta)
                
            # ── Data Preparation ──
            X, y, feature_names = None, None, []
            system_msgs = []
            current_target = "Target"
            
            if raw_df is not None:
                if source_type == "csv":
                    from components.data_prep_ui import render_preprocessing_card
                    clean_df, target_col, prep_msgs = render_preprocessing_card(raw_df, meta.task)
                    system_msgs.extend(prep_msgs)
                    
                    if not clean_df.empty:
                        if target_col and target_col in clean_df.columns:
                            y = clean_df[target_col].values
                            feat_df = clean_df.drop(columns=[target_col])
                            current_target = target_col
                        else:
                            y = None
                            feat_df = clean_df
                            
                        X = feat_df.values
                        feature_names = feat_df.columns.tolist()
                else:
                    # Synthetic datasets are already preprocessed
                    cols_to_drop = []
                    if 'Target' in raw_df.columns and meta.task in ("classification", "regression"):
                        y = raw_df['Target'].values
                        cols_to_drop.append('Target')
                        current_target = 'Target'
                    else:
                        y = None
                        
                    if '_IsOutlier' in raw_df.columns:
                        st.session_state["current_outliers"] = raw_df['_IsOutlier'].values
                        cols_to_drop.append('_IsOutlier')
                    else:
                        st.session_state["current_outliers"] = None
                        
                    feat_df = raw_df.drop(columns=cols_to_drop)
                    X = feat_df.values
                    feature_names = feat_df.columns.tolist()

                if X is not None:
                    if meta.task == "regression" and X.shape[1] > 1:
                        system_msgs.append("Wielowymiarowa regresja: Do wizualizacji wyników użyto widoku 'Rzeczywiste vs Przewidywane' oraz redukcji wymiarów.")
                    elif meta.task == "classification" and X.shape[1] > 2:
                        system_msgs.append("Problem wielowymiarowy: Aplikacja automatycznie użyła analizy główych składowych (PCA), aby wyświetlić podgląd danych w 2D.")

                st.session_state["current_X"] = X
                st.session_state["current_y"] = y
                st.session_state["current_features"] = feature_names
                st.session_state["current_target"] = current_target

            # ── System Messages Inbox ──
            if system_msgs:
                with st.expander(f"✉️ Komunikaty systemu ({len(system_msgs)})", expanded=False):
                    for msg in system_msgs:
                        st.info(f"{msg}")

            # ── Model Attributes & Sidebar Visualizations (Left Column) ──
            if st.session_state.get("model_ran", False):
                render_model_attributes_card(config, config.model_instance, run_features)
                render_side_visualizations(config, config.model_instance, run_X, run_y, meta.task, run_features)
            else:
                render_model_attributes_skeleton(config)
                render_side_visualizations_skeleton(config, X_available=(X is not None))

            # ── Populate Run Model Button ──
            # Executed after params and data are resolved, so no state lagging
            with run_action_placeholder.container():
                if X is not None:
                    st.caption(f"● Dane: **{X.shape[0]}** próbek, **{X.shape[1]}** cech")
                
                run_btn = st.button("▶ Run Model", type="primary", use_container_width=True, disabled=(X is None))
                
                if run_btn and X is not None:
                    st.session_state["model_ran"] = True
                    if not params:
                        params = {p.name: sklearn_params.get(p.name, p.default) for p in visible_params}
                    st.session_state["last_run_params"] = params.copy()
                    st.session_state["last_run_X"] = X
                    st.session_state["last_run_y"] = y
                    st.session_state["last_run_features"] = feature_names
                    st.session_state["last_run_target"] = current_target
                    st.rerun()

        # ==============================================================
        # RIGHT COLUMN — Results Panel
        # ==============================================================
        with col_results:
            if st.session_state.get("model_ran", False):
                if run_X is not None:
                    r_col1, r_col2 = st.columns([3, 1])
                    with r_col1:
                        st.markdown("### 📊 Wyniki")
                    with r_col2:
                        if st.button("💾 Zapisz Model", use_container_width=True):
                            from core.model_storage import save_model
                            try:
                                save_model(
                                    name=meta.name,
                                    plugin_id=st.session_state.current_model_id,
                                    task=meta.task,
                                    model_instance=config.model_instance,
                                    feature_names=run_features,
                                    target_name=st.session_state.get("last_run_target", "Target"),
                                    user_params=run_params
                                )
                                st.toast("✅ Model został zapisany pomyślnie!")
                            except Exception as e:
                                st.error(f"Nie udało się zapisać modelu: {e}")
                    try:
                        render_results_panel(config, config.model_instance, run_X, run_y, run_params, run_features)
                    except Exception as e:
                        st.error(f"⚠️ Błąd: {e}")
                else:
                    st.info("Kliknij **▶ Run Model** aby zobaczyć wyniki.")
            else:
                current_X = st.session_state.get("current_X")
                current_y = st.session_state.get("current_y")
                current_features = st.session_state.get("current_features", [])
                
                try:
                    render_empty_results_panel(
                        config, 
                        current_X, 
                        current_y, 
                        current_features
                    )
                except Exception as e:
                    st.error(f"⚠️ Błąd renderowania widoku startowego: {e}")

    # ================================================================
    # TAB: History
    # ================================================================
    with tab_history:
        st.markdown("#### 📜 Experiment History")
        st.info("Historia eksperymentów będzie dostępna po uruchomieniu modelu. "
                "Każde uruchomienie zapisze parametry, metryki i wyniki.")
        st.caption("Coming soon in next release.")

    # ================================================================
    # TAB: Settings
    # ================================================================
    with tab_settings:
        _render_settings_tab(config)


def _render_ai_suggestion(meta):
    """Renders contextual suggestion based on model type."""
    suggestions = {
        "classification": '"For sparse datasets, consider switching penalty to \'l1\' (Lasso) to encourage feature selection."',
        "regression": '"Try increasing the number of features or adding polynomial features for better fit."',
        "clustering": '"Experiment with different numbers of clusters using the Elbow method."',
        "dimensionality_reduction": '"Check the cumulative explained variance to choose the right number of components."',
    }
    tip = suggestions.get(meta.task, '"Adjust parameters and observe how metrics change."')

    with st.container(border=True):
        st.markdown("**💡 AI Suggestion**")
        st.caption(tip)


def _render_data_card(meta):
    """Renders the Load Data card. Returns (raw_df, source_type)."""
    
    def _reset_model_ran():
        st.session_state["model_ran"] = False

    raw_df = None
    source_type = None

    # Source selector
    csv_selected = st.radio(
        "source",
        ["📄 Upload CSV file", "🧪 Use sample dataset"],
        horizontal=True,
        label_visibility="collapsed",
        key="data_source_radio",
        on_change=_reset_model_ran
    )

    if csv_selected == "📄 Upload CSV file":
        source_type = "csv"
        # Dropzone visual
        st.markdown("""
        <div class="data-dropzone">
            <div class="data-dz-icon">☁️</div>
            <div class="data-dz-title">Drag & Drop training data</div>
            <div class="data-dz-sub">Supports .csv, .xlsx, and .json formats up to 50MB.</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload", type=["csv", "xlsx", "json"], label_visibility="collapsed")

        if uploaded:
            ext = uploaded.name.split(".")[-1].lower()
            if ext == "csv":
                raw_df = pd.read_csv(uploaded)
            elif ext in ("xlsx", "xls"):
                raw_df = pd.read_excel(uploaded)
            elif ext == "json":
                raw_df = pd.read_json(uploaded)
            else:
                raw_df = pd.read_csv(uploaded)

            st.dataframe(raw_df, use_container_width=True, height=280)

        # Format tags
        st.markdown(
            '<div class="data-formats">'
            '<span class="data-fmt">CSV</span>'
            '<span class="data-fmt">XLS</span>'
            '<span class="data-fmt">JSON</span>'
            '</div>',
            unsafe_allow_html=True
        )

    elif csv_selected == "🧪 Use sample dataset":
        source_type = "sample"
        from core.data_generators import generate_dataset

        dataset_map = {
            "classification": "classification_2d",
            "regression": "regression_2d",
            "clustering": "clustering_2d",
            "dimensionality_reduction": "classification_2d"
        }
        dataset_type = dataset_map.get(meta.task, "classification_2d")

        if "data_seed" not in st.session_state:
            st.session_state.data_seed = 42

        trend_type = "positive"
        n_outliers = 5
        if meta.task == "regression":
            trend_options = {
                "📈 Trend dodatni (Liniowy)": "positive",
                "📉 Trend ujemny (Liniowy)": "negative",
                "🪃 Nieliniowy (Parabola)": "parabolic",
                "🧲 Z elementami odstającymi": "outliers",
                "☁️ Brak korelacji (Chmura)": "random"
            }
            sel_trend = st.selectbox(
                "Kształt danych (trend)", 
                list(trend_options.keys()),
                on_change=_reset_model_ran
            )
            trend_type = trend_options[sel_trend]

            if trend_type == "outliers":
                n_outliers = st.slider("Liczba anomalii (0 = odcięcie)", 0, 20, 5, step=1, on_change=_reset_model_ran)

        sc1, sc2 = st.columns(2)
        with sc1:
            n_samples = st.slider("Liczba próbek", 30, 500, 150, step=10, on_change=_reset_model_ran)
        with sc2:
            noise = st.slider("Poziom szumu", 0.01, 0.5, 0.15, step=0.01, on_change=_reset_model_ran)

        if st.button("🎲 Losuj nowe punkty", use_container_width=True, on_click=_reset_model_ran):
            st.session_state.data_seed += 1

        X, y, is_outlier = generate_dataset(
            dataset_type=dataset_type, 
            n_samples=n_samples, 
            noise=noise,
            random_state=st.session_state.data_seed,
            trend_type=trend_type,
            n_outliers=n_outliers
        )

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        feature_names = [f"X{i+1}" for i in range(X.shape[1])]

        raw_df = pd.DataFrame(X, columns=feature_names)
        if y is not None:
            raw_df["Target"] = y
        if np.any(is_outlier):
            raw_df["_IsOutlier"] = is_outlier

        st.dataframe(raw_df.drop(columns=["_IsOutlier"], errors="ignore"), use_container_width=True, height=280)

    return raw_df, source_type



if __name__ == "__main__":
    main()
