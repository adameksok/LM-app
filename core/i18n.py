"""
i18n — Internationalization module for bilingual support (PL/EN).

Usage:
    from core.i18n import t, render_language_selector

    st.button(t("nav.models"))
    render_language_selector()  # place in sidebar
"""

import json
import streamlit as st
from pathlib import Path

_TRANSLATIONS_DIR = Path(__file__).parent.parent / "translations"
_cache: dict = {}


def _load(lang: str) -> dict:
    if lang not in _cache:
        path = _TRANSLATIONS_DIR / f"{lang}.json"
        _cache[lang] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _cache[lang]


def t(key: str) -> str:
    """Return translated string for current language. Falls back to Polish, then to the key itself."""
    lang = st.session_state.get("language", "pl")
    for attempt_lang in ([lang, "pl"] if lang != "pl" else ["pl"]):
        data = _load(attempt_lang)
        val = data
        for part in key.split("."):
            val = val.get(part) if isinstance(val, dict) else None
        if val is not None:
            return str(val)
    return key


def inject_uploader_translations() -> None:
    """
    Overrides Streamlit's built-in (always-English) file uploader texts
    by hiding originals with font-size:0 and injecting translated strings
    via CSS ::after pseudo-elements.
    """
    drag = t("file_uploader.drag_here")
    limit = t("file_uploader.limit_text")
    browse = t("file_uploader.browse_files")

    st.markdown(f"""
    <style>
    /* ── Hide original English strings ── */
    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        font-size: 0 !important;
        line-height: 0 !important;
        display: block;
    }}
    /* ── Inject "drag and drop" translation ── */
    [data-testid="stFileUploaderDropzoneInstructions"] small:first-of-type::after {{
        content: "{drag}";
        font-size: 14px;
        line-height: 1.5;
        color: #31333F;
    }}
    /* ── Inject limit/format translation ── */
    [data-testid="stFileUploaderDropzoneInstructions"] small:last-of-type::after {{
        content: "{limit}";
        font-size: 12px;
        line-height: 1.5;
        color: rgba(49, 51, 63, 0.6);
    }}
    /* ── Translate "Browse files" button ── */
    [data-testid="stFileUploaderDropzone"] button {{
        color: transparent !important;
        position: relative;
    }}
    [data-testid="stFileUploaderDropzone"] button::after {{
        content: "{browse}";
        color: #31333F;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
        white-space: nowrap;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_language_selector() -> None:
    """Renders a language selectbox in the current sidebar context."""
    if "language" not in st.session_state:
        st.session_state.language = "pl"

    options = ["Polski", "English"]
    current_idx = 0 if st.session_state.language == "pl" else 1

    selected = st.selectbox(
        "🌐 Language",
        options,
        index=current_idx,
        key="language_selector"
    )
    new_lang = "pl" if selected == "Polski" else "en"
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()
