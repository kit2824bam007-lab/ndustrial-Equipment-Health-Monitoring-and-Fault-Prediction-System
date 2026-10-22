import numpy as np
import pandas as pd

def create_machine_aware_sequences(df, feature_columns, target_column=None, sequence_length=30, machine_id_column=None):
    """
    Generate sliding time-series 3D sequences (N, sequence_length, num_features).
    CRITICAL RULE: If machine_id_column exists, sequences are created INDEPENDENTLY per machine group.
    Sequences NEVER cross machine boundaries!
    """
    if len(df) < sequence_length:
        raise ValueError(f"Total dataset records ({len(df)}) is less than selected sequence length ({sequence_length}). Please upload a larger dataset or reduce sequence length.")

    X_windows_list = []
    y_windows_list = []

    # Check if Machine ID column is present and valid
    if machine_id_column and machine_id_column in df.columns:
        machine_groups = df.groupby(machine_id_column)
    else:
        # Single implicit machine group
        machine_groups = [('single_group', df)]

    total_valid_machines = 0

    for machine_id, group_df in machine_groups:
        n_rows = len(group_df)
        if n_rows < sequence_length:
            # Skip machine with insufficient history to build a single sequence window
            continue

        total_valid_machines += 1
        feature_matrix = group_df[feature_columns].values
        
        label_array = None
        if target_column and target_column in group_df.columns:
            label_array = group_df[target_column].values

        # Sliding window within this machine's timeline
        for i in range(n_rows - sequence_length + 1):
            window = feature_matrix[i : i + sequence_length]
            X_windows_list.append(window)

            if label_array is not None:
                # Assign target label from final time step of sliding window
                y_windows_list.append(label_array[i + sequence_length - 1])

    if not X_windows_list:
        raise ValueError(f"No valid sequences could be generated. Check that individual machine histories have at least {sequence_length} rows.")

    X_array = np.array(X_windows_list)
    y_array = np.array(y_windows_list) if y_windows_list else None

    return {
        'X': X_array,
        'y': y_array,
        'num_sequences': len(X_array),
        'valid_machines_count': total_valid_machines
    }
