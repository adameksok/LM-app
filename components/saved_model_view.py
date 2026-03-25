import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.model_storage import load_model
from core.model_outputs import build_equation

def render_saved_model_view():
    if "current_saved_model_id" not in st.session_state:
        st.session_state.current_view = "dashboard"
        st.rerun()

    model_id = st.session_state.current_saved_model_id
    try:
        model_data = load_model(model_id)
    except Exception as e:
        st.error(f"Nie można załadować modelu: {e}")
        if st.button("← Powrót do kokpitu"):
            st.session_state.current_view = "dashboard"
            st.rerun()
        return

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <h2>ML Insights</h2>
            <div class="sub">Insight Architect</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        if st.button("🏠  Powrót do menu (Modele)", use_container_width=True):
            st.session_state.current_view = "dashboard"
            del st.session_state.current_saved_model_id
            st.rerun()

    meta_name = model_data["name"]
    task = model_data["task"]
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    target_name = model_data["target_name"]

    st.markdown(f"## 💾 Walidacja: {meta_name}")
    st.caption(f"Zadanie: {task} | Cechy wejściowe: {len(feature_names)}")

    # Model info / Equation
    with st.container(border=True):
        st.markdown("**📐 Parametry modelu**")
        if task == "regression":
            eq = build_equation(model, feature_names, task)
            if eq:
                st.latex(eq.replace("·", r" \cdot ").replace("ŷ", r"\hat{y}"))
            else:
                st.info("Zapisany model regresji.")
        else:
            st.info(f"Zapisany model klasy: {model.__class__.__name__}")

    tab_manual, tab_csv = st.tabs(["✍️ Wprowadź ręcznie", "📁 Wgraj zbiór CSV"])

    with tab_manual:
        st.markdown("### Wprowadź wartości dla pojedynczej próbki")
        cols = st.columns(min(len(feature_names), 4))
        input_data = []

        for i, fname in enumerate(feature_names):
            with cols[i % len(cols)]:
                # default to 0.0
                val = st.number_input(f"{fname}", value=0.0, format="%f", key=f"manual_{fname}")
                input_data.append(val)

        if st.button("🔮 Oblicz Predykcję", type="primary"):
            X_single = np.array([input_data])
            try:
                pred = model.predict(X_single)[0]
                st.success(f"**Wynik predykcji ({target_name}):** {pred:.4f}")
            except Exception as e:
                st.error(f"Błąd podczas predykcji: {e}")

    with tab_csv:
        st.markdown("### Walidacja na nowym zbiorze danych")
        st.info(f"Plik CSV musi zawierać następujące kolumny cech: {', '.join(feature_names)}")

        uploaded = st.file_uploader("Wgraj plik CSV z danymi testowymi", type=["csv"], key="val_uploader")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                missing = [col for col in feature_names if col not in df.columns]
                
                if missing:
                    st.error(f"Brakuje wymaganych kolumn w wgranym pliku: {', '.join(missing)}")
                else:
                    X_val = df[feature_names].values
                    preds = model.predict(X_val)
                    df["Predykcja " + target_name] = preds
                    
                    st.success("Wyliczono predykcje pomyślnie!")

                    if target_name in df.columns:
                        # Obliczanie błędu i wyświetlanie wykresu (Rzeczywiste vs Przewidywane)
                        st.markdown("#### Porównanie: Rzeczywiste vs Przewidywane")
                        y_val = df[target_name].values
                        
                        from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score
                        
                        mcol1, mcol2 = st.columns(2)
                        if task in ["regression"]:
                            rmse = root_mean_squared_error(y_val, preds)
                            r2 = r2_score(y_val, preds)
                            mcol1.metric("RMSE", f"{rmse:.4f}")
                            mcol2.metric("R² Score", f"{r2:.4f}")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=y_val, y=preds, mode='markers', name='Predykcje'))
                            
                            min_v = min(np.min(y_val), np.min(preds))
                            max_v = max(np.max(y_val), np.max(preds))
                            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines', name='Idealne', line=dict(dash='dash', color='red')))
                            
                            fig.update_layout(xaxis_title="Wartość Rzeczywista", yaxis_title="Predykcja", template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif task == "classification":
                            acc = accuracy_score(y_val, preds)
                            mcol1.metric("Accuracy", f"{acc:.4f}")
                    
                    st.markdown("#### Wyniki")
                    st.dataframe(df, use_container_width=True, height=250)
            except Exception as e:
                st.error(f"Błąd podczas analizy pliku: {e}")

