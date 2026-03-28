import streamlit as st


def init_session_state():
    """Initializes default session state values."""
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = None
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "saved_states" not in st.session_state:
        st.session_state.saved_states = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False
