import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_loader import inspect_company_csv, clean_and_sort_data
from sequence_generator import create_machine_aware_sequences

# Re-export for compatibility
inspect_csv = inspect_company_csv

def preview_dataset_preparation(file_path, target_column, feature_columns, sequence_length=30, test_size=0.15, val_size=0.15, machine_id_column=None, timestamp_column=None):
    """
    Preview dataset preparation without training:
    Calculates total rows, total generated 3D sequences, unique target classes,
    class distribution, and sample split counts (train / val / test).
    """
    df_raw = pd.read_csv(file_path)
    
    if target_column not in df_raw.columns:
        raise ValueError(f"Selected target column '{target_column}' does not exist in dataset.")

    missing_features = [f for f in feature_columns if f not in df_raw.columns]
    if missing_features:
        raise ValueError(f"Selected feature column(s) {missing_features} missing from dataset.")

    # Clean and sort data
    df_clean = clean_and_sort_data(
        df=df_raw,
        timestamp_col=timestamp_column,
        machine_id_col=machine_id_column,
        feature_columns=feature_columns
    )

    # Label class distribution
    raw_labels = df_clean[target_column].astype(str).values
    unique_classes, class_counts_arr = np.unique(raw_labels, return_counts=True)
    class_distribution = {str(k): int(v) for k, v in zip(unique_classes, class_counts_arr)}

    if len(unique_classes) < 2:
        raise ValueError(f"Target column '{target_column}' contains only {len(unique_classes)} class ('{unique_classes[0]}'). At least two distinct fault classes are required.")

    # Generate machine-aware sequences
    seq_res = create_machine_aware_sequences(
        df=df_clean,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
        machine_id_column=machine_id_column
    )

    total_sequences = seq_res['num_sequences']
    
    # Chronological split sizes
    n_test = int(total_sequences * test_size)
    n_val = int(total_sequences * val_size)
    n_train = total_sequences - n_test - n_val

    if n_train <= 0:
        raise ValueError("Dataset or sequence count is too small for configured train/validation/test split ratios.")

    return {
        'filename': pd.Series([file_path]).apply(lambda x: x.split('\\')[-1].split('/')[-1]).values[0],
        'total_raw_rows': len(df_raw),
        'clean_rows': len(df_clean),
        'removed_duplicates': len(df_raw) - len(df_clean),
        'num_features': len(feature_columns),
        'sequence_length': sequence_length,
        'total_sequences': total_sequences,
        'valid_machines_count': seq_res['valid_machines_count'],
        'num_classes': len(unique_classes),
        'class_names': [str(c) for c in unique_classes],
        'class_distribution': class_distribution,
        'train_samples': n_train,
        'val_samples': n_val,
        'test_samples': n_test,
        'test_ratio_pct': round(test_size * 100, 1),
        'val_ratio_pct': round(val_size * 100, 1),
        'train_ratio_pct': round((1.0 - test_size - val_size) * 100, 1)
    }
