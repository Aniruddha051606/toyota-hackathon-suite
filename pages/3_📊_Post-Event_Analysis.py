import streamlit as st
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Post-Event Analysis",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading Function ---
@st.cache_data  # Cache the heavy data processing
def load_all_lap_data(csv_file_path, 
                      lap_reset_threshold_meters=-3000, 
                      min_lap_time_seconds=60):
    """
    Loads and processes the entire race file to extract all valid lap times.
    """
    try:
        cols_to_use = ['timestamp', 'telemetry_name', 'telemetry_value']
        df = pd.read_csv(csv_file_path, usecols=cols_to_use)

        df_wide = df.pivot_table(index='timestamp', 
                                 columns='telemetry_name', 
                                 values='telemetry_value',
                                 aggfunc='mean').reset_index()

        df_wide['timestamp_dt'] = pd.to_datetime(df_wide['timestamp'], errors='coerce')
        df_wide = df_wide.sort_values(by='timestamp_dt')
        df_wide['time_sec'] = (df_wide['timestamp_dt'].astype(int) / 10**9)

        telemetry_cols = ['Laptrigger_lapdist_dls'] # Only need this for lap detection
        
        for col in telemetry_cols:
            if col in df_wide.columns:
                df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')

        df_wide[telemetry_cols] = df_wide[telemetry_cols].ffill()
        df_valid = df_wide.dropna(subset=['time_sec', 'Laptrigger_lapdist_dls']).copy()

        df_valid['dist_diff'] = df_valid['Laptrigger_lapdist_dls'].diff()
        lap_crossings = df_valid[df_valid['dist_diff'] < lap_reset_threshold_meters]
        crossing_timestamps_sec = lap_crossings['time_sec'].values
        
        if len(crossing_timestamps_sec) < 2:
            st.error("Not enough lap crossings detected to analyze.")
            return None

        lap_durations = np.diff(crossing_timestamps_sec)
        valid_laps_mask = lap_durations > min_lap_time_seconds
        
        if not np.any(valid_laps_mask):
            st.error(f"No valid laps found with a duration > {min_lap_time_seconds}s.")
            return None
            
        valid_lap_durations = lap_durations[valid_laps_mask]
        
        # Add a "Lap Number"
        lap_data = pd.DataFrame({
            'Lap Number': range(1, len(valid_lap_durations) + 1),
            'Lap Time (s)': valid_lap_durations
        })
        
        return lap_data

    except FileNotFoundError:
        st.error(f"Error: Main data file not found at {csv_file_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None

# --- Main Page Layout ---
st.title("📊 Post-Event Analysis")
st.markdown("Analyzing every lap from the race to find key moments and driver consistency.")

# Load the data
# We use the full CSV file here, not the small lap files
csv_path = r"C:\Users\Aniruddha\Downloads\barber-motorsports-park\barber\R1_barber_telemetry_data.csv"
all_laps = load_all_lap_data(csv_path)

if all_laps is not None:
    # --- Key Statistics ---
    st.subheader("Race Summary Statistics")
    
    fastest_lap_time = all_laps['Lap Time (s)'].min()
    fastest_lap_num = all_laps[all_laps['Lap Time (s)'] == fastest_lap_time]['Lap Number'].values[0]
    
    slowest_lap_time = all_laps['Lap Time (s)'].max()
    slowest_lap_num = all_laps[all_laps['Lap Time (s)'] == slowest_lap_time]['Lap Number'].values[0]
    
    avg_lap_time = all_laps['Lap Time (s)'].mean()
    consistency = all_laps['Lap Time (s)'].std()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fastest Lap", f"{fastest_lap_time:.3f}s", f"Lap {fastest_lap_num}")
    col2.metric("Slowest Lap", f"{slowest_lap_time:.3f}s", f"Lap {slowest_lap_num}")
    col3.metric("Average Lap", f"{avg_lap_time:.3f}s")
    col4.metric("Consistency (StdDev)", f"{consistency:+.3f}s", "Lower is better")
    
    st.divider()
    
    # --- Lap Time Chart ---
    st.subheader("Lap Time Consistency")
    st.markdown("This chart shows your lap time over the entire race. Look for a stable, flat line.")
    
    # Set the index to Lap Number for a clean chart
    chart_data = all_laps.set_index('Lap Number')
    
    st.line_chart(chart_data['Lap Time (s)'])

else:
    st.info("Processing race data... This may take a moment.")