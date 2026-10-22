import os
import numpy as np
import pandas as pd

def inspect_company_csv(file_path):
    """
    Inspect an uploaded company machine dataset CSV and return detailed metadata:
    rows, columns, data types, missing counts, duplicate counts, numeric signals,
    datetime columns, machine ID candidates, target label candidates, basic stats,
    and first 10 rows preview.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV dataset: {str(e)}")

    total_rows = len(df)
    total_cols = len(df.columns)
    columns = list(df.columns)
    dtypes_dict = {col: str(df[col].dtype) for col in columns}
    missing_counts = df.isnull().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())

    # Detect data types and candidates
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    # Datetime candidates
    datetime_candidates = []
    for col in columns:
        if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
            datetime_candidates.append(col)

    # Machine ID candidates
    machine_id_candidates = []
    for col in columns:
        col_lower = col.lower()
        if 'machine' in col_lower or 'asset' in col_lower or 'device' in col_lower or 'equipment' in col_lower or 'unit' in col_lower or 'id' in col_lower:
            machine_id_candidates.append(col)

    # Sensor signal candidates (numeric columns excluding index or ID fields)
    sensor_candidates = [c for c in numeric_cols if c.lower() not in ['id', 'sample_index', 'index'] and c not in machine_id_candidates]
    if not sensor_candidates:
        sensor_candidates = numeric_cols

    # Target label candidates
    target_candidates = []
    for col in columns:
        unique_vals = df[col].nunique()
        if 1 < unique_vals <= 50:
            target_candidates.append(col)

    # Basic statistics for numeric sensor columns
    basic_stats = {}
    if numeric_cols:
        stats_df = df[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
        stats_df.rename(columns={'50%': 'median'}, inplace=True)
        basic_stats = stats_df.round(4).to_dict(orient='index')

    # Top 10 rows preview
    preview_data = df.head(10).replace({np.nan: None}).to_dict(orient='records')

    filename = os.path.basename(file_path)

    return {
        'filename': filename,
        'filepath': file_path,
        'total_rows': total_rows,
        'total_columns': total_cols,
        'columns': columns,
        'dtypes': dtypes_dict,
        'missing_counts': missing_counts,
        'duplicate_count': duplicate_count,
        'numeric_columns': sensor_candidates,
        'datetime_candidates': datetime_candidates,
        'machine_id_candidates': machine_id_candidates,
        'target_candidates': target_candidates,
        'basic_stats': basic_stats,
        'preview': preview_data
    }

def clean_and_sort_data(df, timestamp_col=None, machine_id_col=None, feature_columns=None):
    """
    Clean dataset:
    1. Remove duplicate records
    2. Replace infinity with NaN
    3. Impute missing numeric values using forward-fill followed by column median
    4. Sort chronologically per Machine ID group
    """
    clean_df = df.copy()

    # 1. Remove duplicate records
    clean_df = clean_df.drop_duplicates()

    # 2. Convert infinity to NaN
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)

    # 3. Numeric imputation
    if feature_columns:
        for col in feature_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
                # Forward fill time series data
                clean_df[col] = clean_df[col].ffill()
                # Median fallback
                median_val = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # 4. Sort by timestamp and machine_id
    sort_cols = []
    if machine_id_col and machine_id_col in clean_df.columns:
        sort_cols.append(machine_id_col)
    if timestamp_col and timestamp_col in clean_df.columns:
        # Convert timestamp to datetime if string
        clean_df['_parsed_time'] = pd.to_datetime(clean_df[timestamp_col], errors='coerce')
        sort_cols.append('_parsed_time')

    if sort_cols:
        clean_df = clean_df.sort_values(by=sort_cols).reset_index(drop=True)
        if '_parsed_time' in clean_df.columns:
            clean_df.drop(columns=['_parsed_time'], inplace=True)

    return clean_df
