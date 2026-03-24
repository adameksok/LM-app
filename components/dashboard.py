import streamlit as st
import shutil
from pathlib import Path
from core.plugin_engine import get_plugin_engine
from core.plugin_interface import PluginConfig

MODELS_DIR = Path(__file__).parent.parent / "models"

TASK_ICONS = {
    "classification": "🎯",
    "regression": "📈",
    "clustering": "🔄",
    "dimensionality_reduction": "📊"
}

TASK_LABELS = {
    "classification": "Klasyfikacja",
    "regression": "Regresja",
    "clustering": "Klasteryzacja",
    "dimensionality_reduction": "Redukcja wymiarów"
}

TASK_COLORS = {
    "classification": "#E65100",
    "regression": "#1565C0",
    "clustering": "#2E7D32",
    "dimensionality_reduction": "#6A1B9A"
}


def _inject_dashboard_css():
    st.markdown("""
    <style>
    /* ── Dashboard header ── */
    .dash-header {
        padding: 0 0 8px 0;
    }
    .dash-header .label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #888;
        margin-bottom: 2px;
    }
    .dash-header h2 {
        margin: 0;
        font-size: 22px;
        font-weight: 700;
        color: #1A1A2E;
    }

    /* ── Model card ── */
    .model-card {
        background: #fff;
        border: 1px solid #E8ECF0;
        border-radius: 12px;
        padding: 20px;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: box-shadow 0.2s, transform 0.15s;
        position: relative;
    }
    .model-card:hover {
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }

    /* Top row: icon + badge */
    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 14px;
    }
    .card-icon {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .card-badge {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 10px;
        border-radius: 999px;
    }

    /* Card body */
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #1A1A2E;
        margin-bottom: 6px;
    }
    .card-desc {
        font-size: 12.5px;
        color: #666;
        line-height: 1.45;
        margin-bottom: 14px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Bottom row */
    .card-bottom {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: #999;
        border-top: 1px solid #F0F0F0;
        padding-top: 10px;
    }
    .card-trend {
        font-size: 18px;
        cursor: pointer;
    }

    /* ── Add-model card ── */
    .add-card {
        background: linear-gradient(135deg, #1976D2, #1565C0);
        border: none;
        border-radius: 12px;
        padding: 24px;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .add-card:hover {
        box-shadow: 0 8px 28px rgba(25,118,210,0.3);
        transform: translateY(-2px);
    }
    .add-icon {
        width: 48px;
        height: 48px;
        border: 2px solid rgba(255,255,255,0.5);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        color: #fff;
        margin-bottom: 12px;
    }
    .add-title {
        font-size: 16px;
        font-weight: 700;
        color: #fff;
    }
    .add-subtitle {
        font-size: 12px;
        color: rgba(255,255,255,0.7);
        margin-top: 4px;
    }

    /* ── Fix Streamlit containers used for tiles ── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_dashboard():
    """Renders the model selection dashboard with a 4-column tile grid."""

    _inject_dashboard_css()

    st.markdown("""
        <div class="dash-header">
            <div class="label">Your Models</div>
            <h2>Analytical Library</h2>
        </div>
    """, unsafe_allow_html=True)

    engine = get_plugin_engine()
    plugins = engine.discover_plugins()

    # 4-column grid
    all_items = list(plugins.items())
    cols = st.columns(4, gap="medium")

    for i, (plugin_id, config) in enumerate(all_items):
        with cols[i % 4]:
            _render_tile(plugin_id, config)

    # "Dodaj Model" tile in next available column
    add_col_idx = len(all_items) % 4
    with cols[add_col_idx]:
        _render_add_tile()


def _render_tile(plugin_id: str, config: PluginConfig):
    """Renders a single model card tile."""

    meta = config.metadata
    task = meta.task
    color = TASK_COLORS.get(task, "#1565C0")
    icon = TASK_ICONS.get(task, "📊")
    task_label = TASK_LABELS.get(task, task)

    # Light color variants
    bg_light = color + "18"  # 10% opacity hex

    badge_html = f'<span class="card-badge" style="background:{bg_light}; color:{color};">READY</span>'

    card_html = f"""
    <div class="model-card">
        <div class="card-top">
            <div class="card-icon" style="background:{bg_light}; color:{color};">{icon}</div>
            {badge_html}
        </div>
        <div>
            <div class="card-title">{meta.name or plugin_id}</div>
            <div class="card-desc">{meta.description or task_label}</div>
        </div>
        <div class="card-bottom">
            <span>{task_label}</span>
            <span class="card-trend" style="color:{color};">📈</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    bcol1, bcol2 = st.columns([3, 1])
    with bcol1:
        if st.button("Open →", key=f"btn_{plugin_id}", use_container_width=True):
            st.session_state.current_view = "experiment"
            st.session_state.current_model_id = plugin_id
            st.rerun()
    with bcol2:
        if st.button("🗑️", key=f"del_{plugin_id}", help="Disconnect plugin", use_container_width=True):
            plugin_file = MODELS_DIR / f"{plugin_id}.py"
            try:
                if plugin_file.exists():
                    plugin_file.unlink()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Cannot delete: {e}")


def _render_add_tile():
    """Renders the 'Dodaj Model' tile with upload functionality."""

    add_html = """
    <div class="add-card">
        <div class="add-icon">＋</div>
        <div class="add-title">Add Model</div>
        <div class="add-subtitle">Upload .py file</div>
    </div>
    """
    st.markdown(add_html, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a plugin .py file",
        type=["py"],
        key="plugin_uploader",
        label_visibility="collapsed"
    )

    if uploaded is not None:
        dest = MODELS_DIR / uploaded.name
        if dest.exists():
            st.warning(f"⚠️ `{uploaded.name}` already istnieje — zostanie nadpisany.")

        if st.button("Zainstaluj", key="btn_install", use_container_width=True):
            dest.write_bytes(uploaded.getvalue())
            st.cache_resource.clear()
            st.success(f"✅ Zainstalowano `{uploaded.name}`!")
            st.rerun()
