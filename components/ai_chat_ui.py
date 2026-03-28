"""
components/ai_chat_ui.py
AI assistant chat tab UI component.
Reads lang from st.session_state["lang"] — all UI strings via t().
"""

import os
import streamlit as st
from core.i18n_utils import t
from core.plugin_interface import PluginConfig


def render_ai_chat_tab(config: PluginConfig) -> None:
    """Renders the AI assistant chat tab for the experiment view."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    lang = st.session_state.get("lang", "en")

    # ----------------------------------------------------------------
    # Guard: API key missing
    # ----------------------------------------------------------------
    if not api_key:
        st.warning(t("assistant.no_api_key"))
        st.chat_input(t("assistant.input_placeholder"), disabled=True, key="chat_input_no_key")
        return

    # ----------------------------------------------------------------
    # Import AI logic (lazy — only when tab is active)
    # ----------------------------------------------------------------
    from core.ai_assistant import (
        get_gemini_client,
        build_model_context,
        build_welcome_message,
        validate_question_local,
        get_assistant_response,
    )

    # ----------------------------------------------------------------
    # Gemini client (cached in session_state to avoid re-init)
    # ----------------------------------------------------------------
    if "gemini_client" not in st.session_state:
        try:
            st.session_state["gemini_client"] = get_gemini_client(api_key)
        except Exception as e:
            st.error(f"{t('assistant.error_generic')}: {e}")
            return

    client = st.session_state["gemini_client"]

    # ----------------------------------------------------------------
    # Context banner — always reflects current model state
    # ----------------------------------------------------------------
    meta = config.metadata
    model_ran = st.session_state.get("model_ran", False)
    run_X = st.session_state.get("last_run_X")

    if model_ran and run_X is not None:
        samples_label = "próbek" if lang == "pl" else "samples"
        trained_label = "wytrenowany" if lang == "pl" else "trained"
        st.info(f"🤖  **{meta.name}** · {run_X.shape[0]} {samples_label} · {trained_label}")
    else:
        st.info(t("assistant.model_not_trained_banner"))

    # ----------------------------------------------------------------
    # Welcome message — rendered dynamically, NOT stored in chat_history.
    # This ensures it always reflects the actual current model state,
    # even if the model was trained after the tab was first loaded.
    # ----------------------------------------------------------------
    welcome = build_welcome_message(st.session_state, config, lang)
    with st.chat_message("assistant"):
        st.markdown(welcome)

    # ----------------------------------------------------------------
    # Render actual conversation history (user questions + AI responses)
    # ----------------------------------------------------------------
    history = st.session_state.get("chat_history", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ----------------------------------------------------------------
    # Off-topic rejection messages (built locally, no API call)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Chat input
    # ----------------------------------------------------------------
    user_input = st.chat_input(t("assistant.input_placeholder"))

    if user_input and user_input.strip():
        question = user_input.strip()

        # Show user message immediately
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Layer 1: local whitelist
        # Layer 2 (LLM validation) disabled — halves API calls on free tier.
        # Layer 3 system prompt acts as the final off-topic guard.
        is_valid = validate_question_local(question)

        if not is_valid:
            # Off-topic — reject without API call
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": off_topic_msg}
            )
            with st.chat_message("assistant"):
                st.markdown(off_topic_msg)
        else:
            # Layer 3: Main Gemini response
            with st.chat_message("assistant"):
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
                        st.markdown(response)
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        err_msg = _handle_gemini_error(e, lang)
                        st.error(err_msg)
                        # Roll back user message so history stays consistent
                        st.session_state["chat_history"].pop()

        # Rolling window: keep max 20 messages
        if len(st.session_state["chat_history"]) > 20:
            st.session_state["chat_history"] = st.session_state["chat_history"][-20:]

    # ----------------------------------------------------------------
    # Clear history button
    # ----------------------------------------------------------------
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
