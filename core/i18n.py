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
