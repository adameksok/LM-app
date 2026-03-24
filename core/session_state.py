import streamlit as st


def init_session_state():
    """Initializes default session state values."""
    if "current_view" not in st.session_state:
        st.session_state.current_view = "dashboard"
    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = None
    if "saved_states" not in st.session_state:
        st.session_state.saved_states = []
