import streamlit as st
import pandas as pd
<<<<<<< HEAD
import numpy as np
from typing import Tuple, List, Optional
from core.i18n_utils import t
=======
from core.i18n import t
>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145
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
<<<<<<< HEAD
    st.markdown(f"### 🧰 {t('data_prep.title')}")
    
=======
    st.markdown(t("data_prep.header"))

>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145
    # We will work on a copy of the dataframe
    df = raw_df.copy()
    prep_msgs = []

    with st.container(border=True):
<<<<<<< HEAD
        st.markdown(f"**{t('data_prep.cleaning_header')}**")
        
        # 1. MISSING VALUES
        missing_cols = detect_missing_values(df)
        if missing_cols:
            st.markdown(f'<div class="ml-alert ml-alert-warning"><span>⚠️</span><div>{t("data_prep.missing_values_found")} {", ".join(missing_cols)}</div></div>', unsafe_allow_html=True)
            imp_strategy_selection = st.selectbox(
                label=t("data_prep.missing_values_label"),
                options=[
                    (t("data_prep.mv_drop"), "drop"),
                    (t("data_prep.mv_mean"), "mean"),
                    (t("data_prep.mv_median"), "median"),
                    (t("data_prep.mv_mode"), "mode")
                ],
                format_func=lambda x: x[0],
                help=t("data_prep.mv_help")
            )
            
            imp_strategy = imp_strategy_selection[1] # Get the strategy key from the tuple
            
            df, rows_dropped = apply_imputation(df, imp_strategy, missing_cols)
            if imp_strategy == 'drop' and rows_dropped > 0:
                msg = t('data_prep.msg_missing_drop').format(rows_dropped=rows_dropped)
                st.caption(msg)
                prep_msgs.append(msg)
            elif rows_dropped > 0:
                msg = t('data_prep.msg_missing_impute').format(strategy=imp_strategy, cells=rows_dropped)
                st.caption(msg)
                prep_msgs.append(msg)
        else:
            st.markdown(f'<div class="ml-alert ml-alert-success"><span>✅</span><div>{t("data_prep.no_missing_values")}</div></div>', unsafe_allow_html=True)
=======
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
>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145

        # 2. CATEGORICAL ENCODING
        cat_cols = detect_categorical_columns(df)
        if cat_cols:
<<<<<<< HEAD
            st.markdown(f'<div class="ml-alert ml-alert-warning"><span>⚠️</span><div>{t("data_prep.categorical_found")} {", ".join(cat_cols)}</div></div>', unsafe_allow_html=True)
            st.caption(t("data_prep.ml_numeric_data_req"))
            enc_strategy_selection = st.selectbox(
                label=t("data_prep.categorical_label"),
                options=[
                    (t("data_prep.cat_onehot"), "one-hot"),
                    (t("data_prep.cat_label"), "label"),
                    (t("data_prep.cat_drop"), "drop")
                ],
                format_func=lambda x: x[0],
                help=t("data_prep.cat_help")
            )
            
            enc_strategy = enc_strategy_selection[1] # Get the strategy key from the tuple
            
=======
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

>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145
            df = apply_encoding(df, enc_strategy, cat_cols)
            msg = t("data_prep.encoding_msg").format(cols=', '.join(cat_cols), strategy=enc_strategy)
            prep_msgs.append(msg)
        else:
<<<<<<< HEAD
            st.markdown(f'<div class="ml-alert ml-alert-success"><span>✅</span><div>{t("data_prep.all_numeric")}</div></div>', unsafe_allow_html=True)
=======
            st.success(t("data_prep.all_numeric"))
>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145

        # 3. OUTLIERS
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
<<<<<<< HEAD
            outlier_options = [
                (t("data_prep.outlier_none"), "none"),
                (t("data_prep.outlier_zscore"), "z-score"),
                (t("data_prep.outlier_iqr"), "iqr")
            ]
            
            outlier_selection = st.selectbox(
                label=t("data_prep.outlier_strategy_label"),
                options=outlier_options,
                format_func=lambda x: x[0]
            )
            out_strategy = outlier_selection[1]
            
=======
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

>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145
            df, rows_dropped = remove_outliers(df, out_strategy, numeric_cols)
            if out_strategy != 'none' and rows_dropped > 0:
                msg = t("data_prep.outlier_msg").format(strategy=out_strategy, rows=rows_dropped)
                st.caption(msg)
                prep_msgs.append(msg)

    # TARGET & FEATURE SELECTION
    target_col = None
    if task in ("classification", "regression") and not df.empty:
<<<<<<< HEAD
        st.markdown(f"**🎯 {t('data_prep.feature_selection_header')}**")
=======
        st.markdown(t("data_prep.feature_selection"))
>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145
        cols_list = list(df.columns)

        # Default target logic: try 'Target' or last column
        default_index = len(cols_list) - 1
        if 'Target' in cols_list:
            default_index = cols_list.index('Target')
        elif 'Class' in cols_list:
            default_index = cols_list.index('Class')

        target_col = st.selectbox(
<<<<<<< HEAD
            t("data_prep.target_col_label"),
=======
            t("data_prep.target_variable"),
>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145
            cols_list,
            index=default_index,
            key="prep_target_col"
        )

        available_features = [c for c in cols_list if c != target_col]
        selected_features = st.multiselect(
<<<<<<< HEAD
            t("data_prep.features_col_label"),
=======
            t("data_prep.input_features"),
>>>>>>> 98a778758b754d45ad2d5739f6cd43d378dc5145
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
