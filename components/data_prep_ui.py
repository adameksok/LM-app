import streamlit as st
import pandas as pd
from core.data_prep import (
    detect_missing_values,
    detect_categorical_columns,
    apply_imputation,
    apply_encoding,
    remove_outliers
)

def render_preprocessing_card(raw_df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str, list]:
    """
    Renders the data preparation card. 
    Returns the cleaned dataframe, the selected target column name, and a list of structural messages.
    """
    st.markdown("---")
    st.markdown("### 🧰 Data Preparation")
    
    # We will work on a copy of the dataframe
    df = raw_df.copy()
    prep_msgs = []

    
    with st.container(border=True):
        st.markdown("**Data Cleaning Steps**")
        
        # 1. MISSING VALUES
        missing_cols = detect_missing_values(df)
        if missing_cols:
            st.warning(f"⚠️ Missing values found in columns: {', '.join(missing_cols)}")
            imp_strategy_name = st.selectbox(
                "Missing values strategy:",
                options=[
                    "Drop rows with missing values",
                    "Impute with Mean",
                    "Impute with Median",
                    "Impute with Mode"
                ]
            )
            
            # Map UI selection to strategy key
            strategy_map = {
                "Drop rows with missing values": "drop",
                "Impute with Mean": "mean",
                "Impute with Median": "median",
                "Impute with Mode": "mode"
            }
            imp_strategy = strategy_map[imp_strategy_name]
            
            df, rows_dropped = apply_imputation(df, imp_strategy, missing_cols)
            if imp_strategy == 'drop' and rows_dropped > 0:
                msg = f"Usuwanie braków: Odrzucono {rows_dropped} wierszy z powodu wartości Null."
                st.caption(msg)
                prep_msgs.append(msg)
            elif rows_dropped > 0:
                msg = f"Wypełnianie braków: Zastosowano metodę '{imp_strategy}' dla {rows_dropped} komórek."
                st.caption(msg)
                prep_msgs.append(msg)
        else:
            st.success("✅ No missing values found.")

        # 2. CATEGORICAL ENCODING
        cat_cols = detect_categorical_columns(df)
        if cat_cols:
            st.warning(f"⚠️ Categorical text found in columns: {', '.join(cat_cols)}")
            st.caption("ML models require numeric data.")
            enc_strategy_name = st.selectbox(
                "Encoding strategy:",
                options=[
                    "One-Hot Encoding (Creates 0/1 dummy columns)",
                    "Label Encoding (Assigns integers 0, 1, 2...)",
                    "Drop categorical columns"
                ]
            )
            
            enc_map = {
                "One-Hot Encoding (Creates 0/1 dummy columns)": "one-hot",
                "Label Encoding (Assigns integers 0, 1, 2...)": "label",
                "Drop categorical columns": "drop"
            }
            enc_strategy = enc_map[enc_strategy_name]
            
            df = apply_encoding(df, enc_strategy, cat_cols)
            msg = f"Kodowanie tekstów: Kolumny ({', '.join(cat_cols)}) zmodyfikowano używając strategii '{enc_strategy}'."
            prep_msgs.append(msg)
        else:
            st.success("✅ All columns are numeric.")

        # 3. OUTLIERS (Only relevant for numeric continuous data, but we allow applying it generally to numeric cols)
        # Assuming the remaining columns are numeric
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            outlier_strategy_name = st.selectbox(
                "Outlier removal strategy:",
                options=[
                    "None (Keep original data)",
                    "Z-Score (Remove > 3 standard deviations)",
                    "IQR (Interquartile Range)"
                ]
            )
            
            out_map = {
                "None (Keep original data)": "none",
                "Z-Score (Remove > 3 standard deviations)": "z-score",
                "IQR (Interquartile Range)": "iqr"
            }
            out_strategy = out_map[outlier_strategy_name]
            
            df, rows_dropped = remove_outliers(df, out_strategy, numeric_cols)
            if out_strategy != 'none' and rows_dropped > 0:
                msg = f"Czyszczenie anomalii: Algorytm '{out_strategy}' odrzucił {rows_dropped} wartości odstających."
                st.caption(msg)
                prep_msgs.append(msg)
                
    # TARGET & FEATURE SELECTION
    target_col = None
    if task in ("classification", "regression") and not df.empty:
        st.markdown("**🎯 Feature Selection**")
        cols_list = list(df.columns)
        
        # Default target logic: try 'Target' or last column
        default_index = len(cols_list) - 1
        if 'Target' in cols_list:
            default_index = cols_list.index('Target')
        elif 'Class' in cols_list:
            default_index = cols_list.index('Class')
            
        target_col = st.selectbox(
            "Target Variable (y):",
            cols_list,
            index=default_index,
            key="prep_target_col"
        )
        
        available_features = [c for c in cols_list if c != target_col]
        selected_features = st.multiselect(
            "Input Features (X):",
            options=available_features,
            default=available_features,
            key="prep_features"
        )
        
        if selected_features:
            df = df[selected_features + [target_col]]
        else:
            st.error("Please select at least one input feature.")
            return pd.DataFrame(), None, prep_msgs

    return df, target_col, prep_msgs
