import streamlit as st
import pandas as pd
from core.i18n import t
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
    st.markdown(t("data_prep.header"))

    # We will work on a copy of the dataframe
    df = raw_df.copy()
    prep_msgs = []

    with st.container(border=True):
        st.markdown(t("data_prep.cleaning_steps"))

        # 1. MISSING VALUES
        missing_cols = detect_missing_values(df)
        if missing_cols:
            st.warning(f"{t('data_prep.missing_warning')} {', '.join(missing_cols)}")

            imp_options = [
                t("data_prep.drop_rows"),
                t("data_prep.impute_mean"),
                t("data_prep.impute_median"),
                t("data_prep.impute_mode"),
            ]
            imp_strategies = ["drop", "mean", "median", "mode"]
            imp_idx = st.selectbox(
                t("data_prep.missing_strategy_label"),
                options=range(len(imp_options)),
                format_func=lambda i: imp_options[i]
            )
            imp_strategy = imp_strategies[imp_idx]

            df, rows_dropped = apply_imputation(df, imp_strategy, missing_cols)
            if imp_strategy == 'drop' and rows_dropped > 0:
                msg = t("data_prep.missing_drop_msg").format(rows=rows_dropped)
                st.caption(msg)
                prep_msgs.append(msg)
            elif rows_dropped > 0:
                msg = t("data_prep.missing_impute_msg").format(strategy=imp_strategy, cells=rows_dropped)
                st.caption(msg)
                prep_msgs.append(msg)
        else:
            st.success(t("data_prep.no_missing"))

        # 2. CATEGORICAL ENCODING
        cat_cols = detect_categorical_columns(df)
        if cat_cols:
            st.warning(f"{t('data_prep.cat_warning')} {', '.join(cat_cols)}")
            st.caption(t("data_prep.cat_numeric_required"))

            enc_options = [
                t("data_prep.one_hot"),
                t("data_prep.label_encoding"),
                t("data_prep.drop_cat"),
            ]
            enc_strategies = ["one-hot", "label", "drop"]
            enc_idx = st.selectbox(
                t("data_prep.encoding_label"),
                options=range(len(enc_options)),
                format_func=lambda i: enc_options[i]
            )
            enc_strategy = enc_strategies[enc_idx]

            df = apply_encoding(df, enc_strategy, cat_cols)
            msg = t("data_prep.encoding_msg").format(cols=', '.join(cat_cols), strategy=enc_strategy)
            prep_msgs.append(msg)
        else:
            st.success(t("data_prep.all_numeric"))

        # 3. OUTLIERS
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            out_options = [
                t("data_prep.no_outliers_strategy"),
                t("data_prep.zscore"),
                t("data_prep.iqr"),
            ]
            out_strategies = ["none", "z-score", "iqr"]
            out_idx = st.selectbox(
                t("data_prep.outlier_label"),
                options=range(len(out_options)),
                format_func=lambda i: out_options[i]
            )
            out_strategy = out_strategies[out_idx]

            df, rows_dropped = remove_outliers(df, out_strategy, numeric_cols)
            if out_strategy != 'none' and rows_dropped > 0:
                msg = t("data_prep.outlier_msg").format(strategy=out_strategy, rows=rows_dropped)
                st.caption(msg)
                prep_msgs.append(msg)

    # TARGET & FEATURE SELECTION
    target_col = None
    if task in ("classification", "regression") and not df.empty:
        st.markdown(t("data_prep.feature_selection"))
        cols_list = list(df.columns)

        # Default target logic: try 'Target' or last column
        default_index = len(cols_list) - 1
        if 'Target' in cols_list:
            default_index = cols_list.index('Target')
        elif 'Class' in cols_list:
            default_index = cols_list.index('Class')

        target_col = st.selectbox(
            t("data_prep.target_variable"),
            cols_list,
            index=default_index,
            key="prep_target_col"
        )

        available_features = [c for c in cols_list if c != target_col]
        selected_features = st.multiselect(
            t("data_prep.input_features"),
            options=available_features,
            default=available_features,
            key="prep_features"
        )

        if selected_features:
            df = df[selected_features + [target_col]]
        else:
            st.error(t("data_prep.no_feature_error"))
            return pd.DataFrame(), None, prep_msgs

    return df, target_col, prep_msgs
