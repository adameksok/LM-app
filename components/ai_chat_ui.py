"""
components/ai_chat_ui.py
AI assistant chat tab UI component.
Reads lang from st.session_state["lang"] — all UI strings via t().

Layout: input bar anchored at top, messages grow downward below it.
"""

import os
import streamlit as st
from core.i18n_utils import t
from core.plugin_interface import PluginConfig


def render_ai_chat_tab(config: PluginConfig) -> None:
    """Renders the AI assistant chat tab for the experiment view."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    lang = st.session_state.get("lang", "en")

    # ── Guard: API key missing ──────────────────────────────────────
    if not api_key:
        st.warning(t("assistant.no_api_key"))
        with st.form("chat_disabled_form", clear_on_submit=True):
            col_i, col_b = st.columns([6, 1])
            with col_i:
                st.text_input("q", disabled=True, label_visibility="collapsed",
                              placeholder=t("assistant.input_placeholder"))
            with col_b:
                st.form_submit_button("↑", disabled=True, use_container_width=True)
        return

    # ── Lazy imports ────────────────────────────────────────────────
    from core.ai_assistant import (
        get_gemini_client,
        build_model_context,
        build_welcome_message,
        validate_question_local,
        get_assistant_response,
    )

    # ── Gemini client (cached) ──────────────────────────────────────
    if "gemini_client" not in st.session_state:
        try:
            st.session_state["gemini_client"] = get_gemini_client(api_key)
        except Exception as e:
            st.error(f"{t('assistant.error_generic')}: {e}")
            return
    client = st.session_state["gemini_client"]

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Off-topic rejection copy ────────────────────────────────────
    if lang == "pl":
        off_topic_msg = (
            "Mogę odpowiadać tylko na pytania dotyczące uczenia maszynowego "
            "i wyników Twojego modelu. Spróbuj zapytać np.:\n"
            "- Co oznacza R² w moim modelu?\n"
            "- Dlaczego MAE różni się od RMSE?\n"
            "- Co mi mówi macierz pomyłek?\n"
            "- Jak wpływa parametr K w KNN?"
        )
    else:
        off_topic_msg = (
            "I can only answer questions about machine learning and your model results. "
            "Try asking e.g.:\n"
            "- What does R² mean in my model?\n"
            "- Why does MAE differ from RMSE?\n"
            "- How do I read the confusion matrix?\n"
            "- How does parameter K affect KNN?"
        )

    # ── CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Send button */
    .chat-input-bar div[data-testid="stFormSubmitButton"] > button {
        background-color: #145d91 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        height: 38px !important;
        padding: 0 !important;
        box-shadow: 0 2px 6px rgba(20,93,145,0.2) !important;
    }
    .chat-input-bar div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0f4a73 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(20,93,145,0.25) !important;
    }
    /* Text input */
    .chat-input-bar div[data-testid="stTextInput"] input {
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        background: #fdfdfd !important;
    }
    .chat-input-bar div[data-testid="stTextInput"] input:focus {
        border-color: #145d91 !important;
        box-shadow: 0 0 0 3px rgba(20,93,145,0.1) !important;
    }
    /* Hide form border */
    .chat-input-bar div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # TOP — Input bar (anchored)
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
    with st.form("chat_input_form", clear_on_submit=True):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            user_input = st.text_input(
                "q",
                label_visibility="collapsed",
                placeholder=t("assistant.input_placeholder"),
            )
        with col_btn:
            send_clicked = st.form_submit_button("↑", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # PROCESS input BEFORE rendering messages so new entries appear
    # at the bottom of the history on this very render cycle.
    # ═══════════════════════════════════════════════════════════════
    if send_clicked and user_input and user_input.strip():
        question = user_input.strip()
        st.session_state["chat_history"].append({"role": "user", "content": question})

        is_valid = validate_question_local(question)
        if not is_valid:
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": off_topic_msg}
            )
        else:
            with st.spinner(t("assistant.thinking")):
                try:
                    model_context = build_model_context(st.session_state, config)
                    response = get_assistant_response(
                        question=question,
                        history=st.session_state["chat_history"][:-1],
                        model_context=model_context,
                        lang=lang,
                        client=client,
                    )
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    err_msg = _handle_gemini_error(e, lang)
                    # Roll back user message to keep history consistent
                    st.session_state["chat_history"].pop()
                    st.error(err_msg)

        # Rolling window: keep last 20 messages
        if len(st.session_state["chat_history"]) > 20:
            st.session_state["chat_history"] = st.session_state["chat_history"][-20:]

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # MESSAGES — banner + welcome + history (oldest → newest ↓)
    # ═══════════════════════════════════════════════════════════════

    # Context banner
    meta = config.metadata
    model_ran = st.session_state.get("model_ran", False)
    run_X = st.session_state.get("last_run_X")
    if model_ran and run_X is not None:
        samples_label = "próbek" if lang == "pl" else "samples"
        trained_label = "wytrenowany" if lang == "pl" else "trained"
        st.info(f"🤖  **{meta.name}** · {run_X.shape[0]} {samples_label} · {trained_label}")
    else:
        st.info(t("assistant.model_not_trained_banner"))

    # Welcome message (dynamic, not stored in history)
    welcome = build_welcome_message(st.session_state, config, lang)
    with st.chat_message("assistant"):
        st.markdown(welcome)

    # Conversation history — chronological order, newest at bottom
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Clear button ────────────────────────────────────────────────
    st.markdown("---")
    if st.button(t("assistant.clear_history"), key="btn_clear_chat"):
        st.session_state["chat_history"] = []
        st.toast(t("assistant.clear_confirm"))
        st.rerun()


def _handle_gemini_error(e: Exception, lang: str) -> str:
    """Maps Gemini exceptions to user-friendly i18n messages."""
    err_str = str(e).lower()
    if "timeout" in err_str or "deadline" in err_str:
        return t("assistant.error_timeout")
    if "quota" in err_str or "resource" in err_str or "429" in err_str:
        return t("assistant.error_quota")
    return t("assistant.error_generic")
