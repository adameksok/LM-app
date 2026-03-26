import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.model_storage import load_model
from core.i18n_utils import t
from core.model_outputs import build_equation

def render_saved_model_view():
    if "current_saved_model_id" not in st.session_state:
        st.session_state.page = "dashboard"
        st.rerun()

    model_id = st.session_state.current_saved_model_id
    model_data = load_model(model_id)
    meta_name = model_data.get("name", "Model")
    task = model_data.get("task", "classification")
    model = model_data.get("model")
    feature_names = model_data.get("feature_names", [])
    target_name = model_data.get("target_name", "Target")

    st.markdown(f"## 💾 {t('saved_model.title')}: {meta_name}")
    st.caption(f"{t('saved_model.task_label')}: {task} | {t('saved_model.features_label')}: {len(feature_names)}")

    # Model info / Equation
    with st.container(border=True):
        st.markdown(f"**📐 {t('saved_model.params_header')}**")
        if task == "regression":
            eq = build_equation(model, feature_names, task)
            if eq:
                st.latex(eq.replace("·", r" \cdot ").replace("ŷ", r"\hat{y}"))
            else:
                st.info(t("saved_model.info_regression"))
        else:
            st.info(f"{t('saved_model.saved_class')}: {model.__class__.__name__}")

    tab_manual, tab_csv = st.tabs([t("saved_model.tab_manual"), t("saved_model.tab_csv")])

    with tab_manual:
        st.markdown(f"### {t('saved_model.single_predict_header')}")
        cols = st.columns(min(len(feature_names), 4))
        input_data = []

        for i, fname in enumerate(feature_names):
            with cols[i % len(cols)]:
                # default to 0.0
                val = st.number_input(f"{fname}", value=0.0, format="%f", key=f"manual_{fname}")
                input_data.append(val)

        if st.button(t("saved_model.btn_predict"), type="primary"):
            X_single = np.array([input_data])
            try:
                pred = model.predict(X_single)[0]
                st.success(f"**{t('saved_model.predict_result_prefix')} ({target_name}):** {pred:.4f}")
            except Exception as e:
                st.error(f"{t('saved_model.error_prediction')}: {e}")

    with tab_csv:
        st.markdown(f"### {t('saved_model.batch_validation_header')}")
        st.info(f"{t('saved_model.csv_format_info')} {', '.join(feature_names)}")

        uploaded = st.file_uploader(t("saved_model.csv_uploader_label"), type=["csv"], key="val_uploader")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                missing = [col for col in feature_names if col not in df.columns]
                
                if missing:
                    st.error(f"{t('saved_model.missing_columns_error')} {', '.join(missing)}")
                else:
                    X_val = df[feature_names].values
                    preds = model.predict(X_val)
                    df[f"{t('saved_model.prediction_column_prefix')} {target_name}"] = preds
                    
                    st.success(t("saved_model.success_batch"))

                    if target_name in df.columns:
                        # Obliczanie błędu i wyświetlanie wykresu (Rzeczywiste vs Przewidywane)
                        st.markdown(f"#### {t('saved_model.comparison_header')}")
                        y_val = df[target_name].values
                        
                        from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score
                        
                        mcol1, mcol2 = st.columns(2)
                        if task in ["regression"]:
                            rmse = root_mean_squared_error(y_val, preds)
                            r2 = r2_score(y_val, preds)
                            mcol1.metric("RMSE", f"{rmse:.4f}")
                            r2 = r2_score(y_val, preds)
                            mcol1.metric("RMSE", f"{rmse:.4f}")
                            mcol2.metric("R² Score", f"{r2:.4f}")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=y_val, y=preds, mode='markers', name=t('saved_model.trace_predictions')))
                            
                            min_v = min(np.min(y_val), np.min(preds))
                            max_v = max(np.max(y_val), np.max(preds))
                            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name=t('saved_model.trace_ideal'), line=dict(dash='dash', color='red')))
                            
                            fig.update_layout(xaxis_title=t('saved_model.xaxis_actual'), yaxis_title=t('saved_model.yaxis_predicted'), template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif task == "classification":
                            acc = accuracy_score(y_val, preds)
                            mcol1.metric("Accuracy", f"{acc:.4f}")
                    
                    st.markdown(f"#### {t('saved_model.results_header')}")
                    st.dataframe(df, use_container_width=True, height=250)
            except Exception as e:
                st.error(f"{t('saved_model.error_file_analysis')}: {e}")

