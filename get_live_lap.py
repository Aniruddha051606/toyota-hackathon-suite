import pandas as pd
import numpy as np

def find_median_lap(csv_file_path, 
                    lap_reset_threshold_meters=-3000, 
                    min_lap_time_seconds=60):
    """
    Finds the median (average) lap from the telemetry data and saves it.
    """
    
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
            print(f"Not enough lap crossings detected.")
            return None

        lap_durations = np.diff(crossing_timestamps_sec)
        valid_laps_mask = lap_durations > min_lap_time_seconds
        
        if not np.any(valid_laps_mask):
            print(f"--- ERROR ---")
            print(f"No valid laps found with a duration > {min_lap_time_seconds}s.")
            return None
            
        valid_lap_durations = lap_durations[valid_laps_mask]
        
        # --- THIS IS THE MAIN CHANGE ---
        # 10. Find the median (average) lap time
        median_duration = np.median(valid_lap_durations)
        
        # Find the lap duration that is *closest* to the median
        median_lap_duration = valid_lap_durations[
            np.argmin(np.abs(valid_lap_durations - median_duration))
        ]
        
        # Find the index of this lap in the *original* durations array
        median_lap_index = np.where(lap_durations == median_lap_duration)[0][0]
        # --- END CHANGE ---

        start_time_sec = crossing_timestamps_sec[median_lap_index]
        end_time_sec = crossing_timestamps_sec[median_lap_index + 1]

        print(f"\nMedian (average) lap found!")
        print(f"  Lap Time: {median_lap_duration:.3f} seconds")

        live_lap_data = df_valid[
            (df_valid['time_sec'] >= start_time_sec) &
            (df_valid['time_sec'] < end_time_sec)
        ].copy()

        live_lap_data['lap_timestamp'] = live_lap_data['time_sec'] - live_lap_data['time_sec'].min()
        
        print(f"Extracted {len(live_lap_data)} data points for the live lap.")
        
        live_lap_data.to_csv("live_lap.csv", index=False)
        print("Live lap data saved to 'live_lap.csv'")
        
        return live_lap_data

    except FileNotFoundError:
        print(f"--- ERROR: File not found ---")
        return None
    except KeyError as e:
        print(f"--- ERROR: Column not found ---")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- How to use the function ---
csv_path = r"C:\Users\Aniruddha\Downloads\barber-motorsports-park\barber\R1_barber_telemetry_data.csv"
find_median_lap(csv_path)