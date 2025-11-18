import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model(csv_file_path, lap_reset_threshold_meters=-3000, min_lap_time_seconds=60):
    print("Starting model training process...")
    try:
        # 1. Load Data
        cols = ['timestamp', 'telemetry_name', 'telemetry_value']
        print(f"Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path, usecols=cols)
        
        print("Pivoting data... (This takes a moment)")
        df_wide = df.pivot_table(index='timestamp', columns='telemetry_name', values='telemetry_value', aggfunc='mean').reset_index()
        
        df_wide['timestamp_dt'] = pd.to_datetime(df_wide['timestamp'], errors='coerce')
        df_wide = df_wide.sort_values(by='timestamp_dt')
        df_wide['time_sec'] = (df_wide['timestamp_dt'].astype(int) / 10**9)
        
        numeric_cols = ['Laptrigger_lapdist_dls', 'Steering_Angle', 'aps', 'nmot', 'pbrake_f', 'speed']
        for col in numeric_cols:
            if col in df_wide.columns: df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')
            
        df_wide = df_wide.ffill().dropna(subset=['time_sec', 'Laptrigger_lapdist_dls'])
        df_wide['dist_diff'] = df_wide['Laptrigger_lapdist_dls'].diff()
        
        # 2. Identify Laps
        crossing_times = df_wide[df_wide['dist_diff'] < lap_reset_threshold_meters]['time_sec'].values
        if len(crossing_times) < 2: 
            print("Error: Not enough laps.")
            return
        
        durations = np.diff(crossing_times)
        valid_mask = durations > min_lap_time_seconds
        valid_durations = durations[valid_mask]
        starts = crossing_times[:-1][valid_mask]
        ends = crossing_times[1:][valid_mask]
        
        print(f"Found {len(valid_durations)} valid laps. Engineering features...")

        # 3. Feature Engineering
        features = []
        for i in range(len(valid_durations)):
            lap = df_wide[(df_wide['time_sec'] >= starts[i]) & (df_wide['time_sec'] < ends[i])]
            if lap.empty: continue
            features.append({
                'avg_speed': lap['speed'].mean(), 
                'max_speed': lap['speed'].max(),
                'avg_rpm': lap['nmot'].mean(), 
                'max_rpm': lap['nmot'].max(),
                'avg_throttle': lap['aps'].mean(), 
                'percent_full_throttle': (lap['aps'] > 95).mean() * 100,
                'percent_braking': (lap['pbrake_f'] > 5).mean() * 100,
                'avg_steering_angle': lap['Steering_Angle'].abs().mean()
            })
            
        X = pd.DataFrame(features).fillna(0)
        y = valid_durations
        
        # 4. Train
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # 5. Save
        joblib.dump(model, "lap_time_model.pkl")
        joblib.dump(list(X.columns), "model_features.pkl")
        print("✅ Model trained and saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

# --- Run ---
csv_path = "indianapolis/R2_indianapolis_motor_speedway_telemetry.csv"
train_model(csv_path)