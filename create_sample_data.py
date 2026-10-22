import numpy as np
import pandas as pd
import os

# Set seed for reproducible synthetic dataset
np.random.seed(42)

num_samples_per_class = 500
classes = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault']

records = []
timestamp_counter = 0

for cls in classes:
    t = np.linspace(0, 10, num_samples_per_class)
    
    if cls == 'Normal':
        # Baseline low noise sinusoidal vibration
        vib_de = 0.05 * np.sin(2 * np.pi * 30 * t) + np.random.normal(0, 0.02, num_samples_per_class)
        vib_fe = 0.04 * np.cos(2 * np.pi * 30 * t) + np.random.normal(0, 0.015, num_samples_per_class)
        temp = 42.0 + 0.5 * np.sin(t) + np.random.normal(0, 0.2, num_samples_per_class)
        current = 10.1 + np.random.normal(0, 0.1, num_samples_per_class)
        rpm = 1790 + np.random.normal(0, 2.0, num_samples_per_class)

    elif cls == 'Inner Race Fault':
        # High frequency impacts with periodic amplitude modulation (BPFI ~ 150Hz)
        vib_de = 0.35 * np.sin(2 * np.pi * 30 * t) + 0.6 * np.sin(2 * np.pi * 150 * t) * (1 + 0.8 * np.sin(2 * np.pi * 30 * t)) + np.random.normal(0, 0.08, num_samples_per_class)
        vib_fe = 0.25 * np.cos(2 * np.pi * 30 * t) + 0.4 * np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.06, num_samples_per_class)
        temp = 58.5 + 1.2 * t + np.random.normal(0, 0.4, num_samples_per_class)
        current = 12.4 + np.random.normal(0, 0.3, num_samples_per_class)
        rpm = 1775 + np.random.normal(0, 4.0, num_samples_per_class)

    elif cls == 'Outer Race Fault':
        # Sharp high-amplitude periodic impacts (BPFO ~ 105Hz)
        vib_de = 0.25 * np.sin(2 * np.pi * 30 * t) + 0.8 * (np.sin(2 * np.pi * 105 * t) ** 3) + np.random.normal(0, 0.09, num_samples_per_class)
        vib_fe = 0.30 * np.cos(2 * np.pi * 30 * t) + 0.5 * (np.sin(2 * np.pi * 105 * t) ** 3) + np.random.normal(0, 0.07, num_samples_per_class)
        temp = 64.0 + 0.8 * np.sin(t * 2) + np.random.normal(0, 0.5, num_samples_per_class)
        current = 13.8 + np.random.normal(0, 0.4, num_samples_per_class)
        rpm = 1760 + np.random.normal(0, 5.0, num_samples_per_class)

    elif cls == 'Ball Fault':
        # Non-harmonic transient spikes and elevated thermal signature (BSF ~ 70Hz)
        spikes = np.random.choice([0, 1.2, -1.2], size=num_samples_per_class, p=[0.90, 0.05, 0.05])
        vib_de = 0.30 * np.sin(2 * np.pi * 30 * t) + 0.45 * np.sin(2 * np.pi * 70 * t) + spikes + np.random.normal(0, 0.07, num_samples_per_class)
        vib_fe = 0.20 * np.cos(2 * np.pi * 30 * t) + 0.35 * np.sin(2 * np.pi * 70 * t) + 0.5 * spikes + np.random.normal(0, 0.05, num_samples_per_class)
        temp = 53.0 + np.random.normal(0, 0.3, num_samples_per_class)
        current = 11.7 + np.random.normal(0, 0.25, num_samples_per_class)
        rpm = 1782 + np.random.normal(0, 3.0, num_samples_per_class)

    for i in range(num_samples_per_class):
        records.append({
            'sample_index': timestamp_counter,
            'vibration_drive_end': round(float(vib_de[i]), 5),
            'vibration_fan_end': round(float(vib_fe[i]), 5),
            'temperature': round(float(temp[i]), 2),
            'motor_current': round(float(current[i]), 2),
            'rotational_speed': round(float(rpm[i]), 1),
            'fault_label': cls
        })
        timestamp_counter += 1

df = pd.DataFrame(records)

target_path = os.path.join(os.path.dirname(__file__), 'data', 'sample', 'sample_machine_data.csv')
os.makedirs(os.path.dirname(target_path), exist_ok=True)
df.to_csv(target_path, index=False)
print(f"Generated sample dataset with {len(df)} rows at {target_path}")
