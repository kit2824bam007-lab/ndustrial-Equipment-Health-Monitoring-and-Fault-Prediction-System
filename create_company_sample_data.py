import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

machines = ['M001', 'M002', 'M003']
classes = ['Normal', 'Bearing Fault', 'Overheating', 'Motor Fault']
samples_per_machine_class = 250

records = []
start_time = datetime(2026, 9, 1, 8, 0, 0)

for m_id in machines:
    current_time = start_time
    for cls in classes:
        t = np.linspace(0, 10, samples_per_machine_class)
        
        if cls == 'Normal':
            temp = 45.0 + 1.2 * np.sin(t) + np.random.normal(0, 0.3, samples_per_machine_class)
            vib = 0.04 * np.sin(2 * np.pi * 25 * t) + np.random.normal(0, 0.015, samples_per_machine_class)
            current = 10.2 + np.random.normal(0, 0.1, samples_per_machine_class)
            rpm = 1795 + np.random.normal(0, 2.0, samples_per_machine_class)
            pressure = 2.4 + np.random.normal(0, 0.05, samples_per_machine_class)
            load = 65.0 + np.random.normal(0, 1.0, samples_per_machine_class)
            op_hours = np.linspace(100, 150, samples_per_machine_class)

        elif cls == 'Bearing Fault':
            temp = 55.0 + np.random.normal(0, 0.4, samples_per_machine_class)
            vib = 0.35 * np.sin(2 * np.pi * 25 * t) + 0.6 * (np.sin(2 * np.pi * 120 * t) ** 3) + np.random.normal(0, 0.08, samples_per_machine_class)
            current = 12.1 + np.random.normal(0, 0.25, samples_per_machine_class)
            rpm = 1770 + np.random.normal(0, 4.0, samples_per_machine_class)
            pressure = 2.5 + np.random.normal(0, 0.08, samples_per_machine_class)
            load = 72.0 + np.random.normal(0, 1.5, samples_per_machine_class)
            op_hours = np.linspace(500, 550, samples_per_machine_class)

        elif cls == 'Overheating':
            temp = 82.5 + 2.0 * t + np.random.normal(0, 0.6, samples_per_machine_class)
            vib = 0.12 * np.sin(2 * np.pi * 25 * t) + np.random.normal(0, 0.03, samples_per_machine_class)
            current = 14.8 + np.random.normal(0, 0.4, samples_per_machine_class)
            rpm = 1750 + np.random.normal(0, 5.0, samples_per_machine_class)
            pressure = 3.1 + np.random.normal(0, 0.12, samples_per_machine_class)
            load = 88.0 + np.random.normal(0, 2.0, samples_per_machine_class)
            op_hours = np.linspace(800, 850, samples_per_machine_class)

        elif cls == 'Motor Fault':
            temp = 68.0 + np.random.normal(0, 0.5, samples_per_machine_class)
            vib = 0.22 * np.sin(2 * np.pi * 25 * t) + np.random.normal(0, 0.05, samples_per_machine_class)
            current = 18.5 + 2.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.6, samples_per_machine_class)
            rpm = 1720 + 20.0 * np.sin(t) + np.random.normal(0, 6.0, samples_per_machine_class)
            pressure = 2.8 + np.random.normal(0, 0.10, samples_per_machine_class)
            load = 80.0 + np.random.normal(0, 2.5, samples_per_machine_class)
            op_hours = np.linspace(1200, 1250, samples_per_machine_class)

        for i in range(samples_per_machine_class):
            current_time += timedelta(minutes=5)
            records.append({
                'Timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'Machine_ID': m_id,
                'Temperature': round(float(temp[i]), 2),
                'Vibration': round(float(vib[i]), 4),
                'Current': round(float(current[i]), 2),
                'RPM': round(float(rpm[i]), 1),
                'Pressure': round(float(pressure[i]), 2),
                'Load': round(float(load[i]), 1),
                'Operating_Hours': round(float(op_hours[i]), 1),
                'Fault_Status': cls
            })

df = pd.DataFrame(records)

target_path = os.path.join(os.path.dirname(__file__), 'data', 'sample', 'company_machine_data.csv')
os.makedirs(os.path.dirname(target_path), exist_ok=True)
df.to_csv(target_path, index=False)
print(f"Generated realistic company dataset with {len(df)} rows at {target_path}")
