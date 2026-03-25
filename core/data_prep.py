import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def detect_missing_values(df: pd.DataFrame) -> list:
    """Returns a list of column names that contain missing values."""
    return df.columns[df.isnull().any()].tolist()

def detect_categorical_columns(df: pd.DataFrame) -> list:
    """Returns a list of column names that are categorical or strings."""
    return df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

def apply_imputation(df: pd.DataFrame, strategy: str, columns: list) -> tuple[pd.DataFrame, int]:
    """
    Applies imputation to specific columns.
    Returns the cleaned dataframe and the number of rows modified/dropped.
    Strategies: 'drop', 'mean', 'median', 'mode'
    """
    df_clean = df.copy()
    rows_affected = 0
    
    if strategy == 'drop':
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=columns)
        rows_affected = initial_len - len(df_clean)
    else:
        for col in columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    val = df_clean[col].mean()
                elif strategy == 'median':
                    val = df_clean[col].median()
                else: # mode fallback for numeric
                    val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            else:
                # For non-numeric, always use mode
                val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else ""
                
            null_count = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(val)
            rows_affected += null_count

    return df_clean, rows_affected

def apply_encoding(df: pd.DataFrame, strategy: str, columns: list) -> pd.DataFrame:
    """
    Encodes categorical columns.
    Strategies: 'drop', 'one-hot', 'label'
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.drop(columns=columns)
    elif strategy == 'label':
        le = LabelEncoder()
        for col in columns:
            # Handle possible NaNs before encoding
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = le.fit_transform(df_clean[col])
    elif strategy == 'one-hot':
        df_clean = pd.get_dummies(df_clean, columns=columns, drop_first=True, dtype=int)
        
    return df_clean

def remove_outliers(df: pd.DataFrame, strategy: str, columns: list) -> tuple[pd.DataFrame, int]:
    """
    Removes outliers from numeric columns.
    Strategies: 'z-score', 'iqr', 'none'
    Returns the cleaned dataframe and number of rows dropped.
    """
    if strategy == 'none' or not columns:
        return df, 0
        
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    # Filter only numeric columns
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df_clean[c])]
    
    if not numeric_cols:
        return df_clean, 0

    if strategy == 'z-score':
        # Remove rows where any numeric column is > 3 std devs from mean
        for col in numeric_cols:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                z_scores = np.abs((df_clean[col] - mean) / std)
                df_clean = df_clean[z_scores <= 3]
                
    elif strategy == 'iqr':
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    rows_dropped = initial_len - len(df_clean)
    return df_clean, rows_dropped
