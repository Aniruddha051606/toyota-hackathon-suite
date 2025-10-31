import pandas as pd
import numpy as np

def find_ghost_lap(csv_file_path, 
                   lap_reset_threshold_meters=-3000, 
                   min_lap_time_seconds=60):
    print(f"Loading data from {csv_file_path}...")
    
    try:
        cols_to_use = ['timestamp', 'telemetry_name', 'telemetry_value']
        df = pd.read_csv(csv_file_path, usecols=cols_to_use)

        print("Data loaded. Pivoting data... (This may take a moment)")
        df_wide = df.pivot_table(index='timestamp', 
                                 columns='telemetry_name', 
                                 values='telemetry_value',
                                 aggfunc='mean').reset_index()

        print("Data pivoted. Converting timestamps...")

        df_wide['timestamp_dt'] = pd.to_datetime(df_wide['timestamp'], errors='coerce')
        df_wide = df_wide.sort_values(by='timestamp_dt')
        df_wide['time_sec'] = (df_wide['timestamp_dt'].astype(int) / 10**9)

        print("Handling sparse telemetry data...")
        
        telemetry_cols = [
            'Laptrigger_lapdist_dls', 'Steering_Angle', 'VBOX_Lat_Min', 
            'VBOX_Long_Minutes', 'accx_can', 'accy_can', 'aps', 'gear', 
            'nmot', 'pbrake_f', 'pbrake_r', 'speed'
        ]
        
        for col in telemetry_cols:
            if col in df_wide.columns:
                df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')

        df_wide[telemetry_cols] = df_wide[telemetry_cols].ffill()
        df_valid = df_wide.dropna(subset=['time_sec', 'Laptrigger_lapdist_dls']).copy()

        print("Detecting lap crossings...")
        df_valid['dist_diff'] = df_valid['Laptrigger_lapdist_dls'].diff()
        lap_crossings = df_valid[df_valid['dist_diff'] < lap_reset_threshold_meters]
        crossing_timestamps_sec = lap_crossings['time_sec'].values
        
        if len(crossing_timestamps_sec) < 2:
            print(f"--- ERROR ---")
            print(f"Not enough lap crossings detected with threshold {lap_reset_threshold_meters}.")
            return None
        lap_durations = np.diff(crossing_timestamps_sec)
        valid_laps_mask = lap_durations > min_lap_time_seconds
        if not np.any(valid_laps_mask):
            print(f"--- ERROR ---")
            print(f"No valid laps found with a duration > {min_lap_time_seconds}s.")
            return None
        valid_lap_durations = lap_durations[valid_laps_mask]
    
        fastest_duration = np.min(valid_lap_durations)
        
        fastest_lap_index = np.where(lap_durations == fastest_duration)[0][0]
        

        start_time_sec = crossing_timestamps_sec[fastest_lap_index]
        end_time_sec = crossing_timestamps_sec[fastest_lap_index + 1]

        print(f"\nFastest lap found!")
        print(f"  Lap Time: {fastest_duration:.3f} seconds")

        ghost_lap_data = df_valid[
            (df_valid['time_sec'] >= start_time_sec) &
            (df_valid['time_sec'] < end_time_sec)
        ].copy()

        ghost_lap_data['lap_timestamp'] = ghost_lap_data['time_sec'] - ghost_lap_data['time_sec'].min()
        
        print(f"Extracted {len(ghost_lap_data)} data points for the ghost lap.")
        
 
        ghost_lap_data.to_csv("ghost_lap.csv", index=False)
        print("Ghost lap data saved to 'ghost_lap.csv'")
        
        return ghost_lap_data

    except FileNotFoundError:
        print(f"--- ERROR: File not found ---")
        return None
    except KeyError as e:
        print(f"--- ERROR: Column not found ---")
        print(f"A required column ({e}) was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


csv_path = r"C:\Users\Aniruddha\Downloads\barber-motorsports-park\barber\R1_barber_telemetry_data.csv"


ghost_data = find_ghost_lap(csv_path)

if ghost_data is not None:
    print("\n--- Ghost Lap Data (First 5 Rows) ---")
    pd.set_option('display.max_columns', None) 
    print(ghost_data.head())