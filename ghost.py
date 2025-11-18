import pandas as pd
import numpy as np

def find_ghost_lap(csv_file_path, lap_reset_threshold_meters=-3000, min_lap_time_seconds=60):
    print(f"Loading data from {csv_file_path}...")
    try:
        cols = ['timestamp', 'telemetry_name', 'telemetry_value']
        df = pd.read_csv(csv_file_path, usecols=cols)
        print("Pivoting...")
        df_wide = df.pivot_table(index='timestamp', columns='telemetry_name', values='telemetry_value', aggfunc='mean').reset_index()
        
        df_wide['timestamp_dt'] = pd.to_datetime(df_wide['timestamp'], errors='coerce')
        df_wide = df_wide.sort_values(by='timestamp_dt')
        df_wide['time_sec'] = (df_wide['timestamp_dt'].astype(int) / 10**9)
        
        numeric_cols = ['Laptrigger_lapdist_dls', 'Steering_Angle', 'VBOX_Lat_Min', 'VBOX_Long_Minutes', 'aps', 'gear', 'nmot', 'pbrake_f', 'pbrake_r', 'speed']
        for col in numeric_cols:
            if col in df_wide.columns: df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')
            
        df_wide = df_wide.ffill().dropna(subset=['time_sec', 'Laptrigger_lapdist_dls'])
        df_wide['dist_diff'] = df_wide['Laptrigger_lapdist_dls'].diff()
        
        crossing_times = df_wide[df_wide['dist_diff'] < lap_reset_threshold_meters]['time_sec'].values
        if len(crossing_times) < 2: return
        
        durations = np.diff(crossing_times)
        valid_mask = durations > min_lap_time_seconds
        if not np.any(valid_mask): return
        
        valid_durations = durations[valid_mask]
        fastest_time = np.min(valid_durations)
        idx = np.where(durations == fastest_time)[0][0]
        
        start, end = crossing_times[idx], crossing_times[idx+1]
        ghost_lap = df_wide[(df_wide['time_sec'] >= start) & (df_wide['time_sec'] < end)].copy()
        ghost_lap['lap_timestamp'] = ghost_lap['time_sec'] - ghost_lap['time_sec'].min()
        
        ghost_lap.to_csv("ghost_lap.csv", index=False)
        print(f"Ghost lap saved ({fastest_time:.3f}s)")
        
    except Exception as e:
        print(f"Error: {e}")

# --- UPDATED PATH FOR INDY ---
csv_path = "indianapolis/R2_indianapolis_motor_speedway_telemetry.csv"
find_ghost_lap(csv_path)