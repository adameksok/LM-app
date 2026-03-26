import json
import streamlit as st
from pathlib import Path

TRANSLATIONS_FILE = Path(__file__).parent.parent / "translations.json"

@st.cache_data
def load_translations():
    """Loads the translations dictionary from the JSON file."""
    try:
        if not TRANSLATIONS_FILE.exists():
            return {}
        with open(TRANSLATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def t(key, **kwargs):
    """
    Translates a key based on the current session language ('lang').
    Supports nested keys with dot notation (e.g., 'dashboard.title').
    
    Fallback: returns the key itself if the translation is missing.
    """
    if not key:
        return ""
        
    # Get current language from session state (fallback to 'en')
    lang = st.session_state.get("lang", "en")
    
    # Load all translations
    translations = load_translations()
    
    # Get the dictionary for the current language
    t_dict = translations.get(lang, {})
    
    # 1. Try dot-notation lookup (e.g. 'sidebar.title')
    res = t_dict
    parts = str(key).split(".")
    found_dot = False
    if len(parts) > 1:
        for k in parts:
            if isinstance(res, dict) and k in res:
                res = res[k]
            else:
                res = None
                break
        if isinstance(res, str):
            found_dot = True
            
    # 2. Try direct lookup in 'plugins' section if dot-notation failed
    if not found_dot:
        res = t_dict.get("plugins", {}).get(key)
        
    # FALLBACK: If still not found, return the key itself
    if not isinstance(res, str):
        # Last attempt: check if key itself is in t_dict as a flat key
        res = t_dict.get(key)
        if not isinstance(res, str):
            return str(key)
        
    # Apply dynamic formatting if variables are provided
    try:
        return res.format(**kwargs)
    except Exception:
        return res

def format_number(val):
    """Formats a number based on the current language (dot vs comma)."""
    lang = st.session_state.get("lang", "en")
    if isinstance(val, (int, float)):
        if lang == "pl":
            # Polish format: space as thousands separator, comma as decimal
            return f"{val:,.4f}".replace(",", " ").replace(".", ",").replace(" ", "\u00A0")
        return f"{val:,.4f}"
    return str(val)

def format_date(timestamp):
    """Formats a timestamp based on the current language."""
    import datetime
    lang = st.session_state.get("lang", "en")
    dt = datetime.datetime.fromtimestamp(timestamp)
    if lang == "pl":
        return dt.strftime("%d.%m.%Y, %H:%M:%S")
    return dt.strftime("%Y-%m-%d, %H:%M:%S")
