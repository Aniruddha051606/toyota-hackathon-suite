import pandas as pd
import numpy as np
import joblib # Used for saving the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model(csv_file_path, 
                lap_reset_threshold_meters=-3000, 
                min_lap_time_seconds=60):
    """
    Loads the full race CSV, extracts features for every lap,
    trains an ML model, and saves it to a file.
    """
    print("Starting model training process...")
    
    try:
        # --- 1. Load and Process Data (Same as Post-Event) ---
        print("Loading and pivoting data... (This may take 1-2 minutes)")
        cols_to_use = ['timestamp', 'telemetry_name', 'telemetry_value']
        df = pd.read_csv(csv_file_path, usecols=cols_to_use)

        df_wide = df.pivot_table(index='timestamp', 
                                 columns='telemetry_name', 
                                 values='telemetry_value',
                                 aggfunc='mean').reset_index()

        df_wide['timestamp_dt'] = pd.to_datetime(df_wide['timestamp'], errors='coerce')
        df_wide = df_wide.sort_values(by='timestamp_dt')
        df_wide['time_sec'] = (df_wide['timestamp_dt'].astype(int) / 10**9)

        telemetry_cols = [
            'Laptrigger_lapdist_dls', 'Steering_Angle', 'aps', 'gear', 
            'nmot', 'pbrake_f', 'speed'
        ]
        
        for col in telemetry_cols:
            if col in df_wide.columns:
                df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')

        df_wide[telemetry_cols] = df_wide[telemetry_cols].ffill()
        df_valid = df_wide.dropna(subset=['time_sec', 'Laptrigger_lapdist_dls']).copy()

        print("Detecting laps...")
        df_valid['dist_diff'] = df_valid['Laptrigger_lapdist_dls'].diff()
        lap_crossings = df_valid[df_valid['dist_diff'] < lap_reset_threshold_meters]
        crossing_timestamps_sec = lap_crossings['time_sec'].values
        
        if len(crossing_timestamps_sec) < 2:
            print("Error: Not enough laps found to train.")
            return None

        lap_durations = np.diff(crossing_timestamps_sec)
        valid_laps_mask = lap_durations > min_lap_time_seconds
        valid_lap_durations = lap_durations[valid_laps_mask]
        
        # Get the start/end times for all valid laps
        valid_start_times = crossing_timestamps_sec[:-1][valid_laps_mask]
        valid_end_times = crossing_timestamps_sec[1:][valid_laps_mask]

        # --- 2. Feature Engineering ---
        print(f"Found {len(valid_lap_durations)} valid laps. Engineering features...")
        features = []
        targets = []

        for i in range(len(valid_lap_durations)):
            start_t = valid_start_times[i]
            end_t = valid_end_times[i]
            lap_duration = valid_lap_durations[i]
            
            # Get the telemetry data for this one lap
            lap_df = df_valid[
                (df_valid['time_sec'] >= start_t) &
                (df_valid['time_sec'] < end_t)
            ]
            
            if lap_df.empty:
                continue

            # Calculate features for this lap
            lap_features = {
                'avg_speed': lap_df['speed'].mean(),
                'max_speed': lap_df['speed'].max(),
                'avg_rpm': lap_df['nmot'].mean(),
                'max_rpm': lap_df['nmot'].max(),
                'avg_throttle': lap_df['aps'].mean(),
                'percent_full_throttle': (lap_df['aps'] > 95).mean() * 100,
                'percent_braking': (lap_df['pbrake_f'] > 5).mean() * 100,
                'avg_steering_angle': lap_df['Steering_Angle'].abs().mean()
            }
            
            features.append(lap_features)
            targets.append(lap_duration)

        X = pd.DataFrame(features)
        y = np.array(targets)
        
        # Clean up any NaNs from feature calculation
        X = X.fillna(X.mean())

        print("Features engineered. Training model...")
        
        # --- 3. Train the Model ---
        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # --- 4. Evaluate and Save ---
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"\nModel trained!")
        print(f"  Test Mean Absolute Error: {mae:.3f} seconds")
        print(f"  (This means the model's predictions are, on average, {mae:.3f}s off)")

        # Save the trained model
        model_filename = "lap_time_model.pkl"
        joblib.dump(model, model_filename)
        print(f"\nModel successfully saved to '{model_filename}'")
        
        # Save the feature columns for the app
        feature_filename = "model_features.pkl"
        joblib.dump(list(X.columns), feature_filename)
        print(f"Model features saved to '{feature_filename}'")


    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the training ---
csv_path = r"C:\Users\Aniruddha\Downloads\barber-motorsports-park\barber\R1_barber_telemetry_data.csv"
train_model(csv_path)