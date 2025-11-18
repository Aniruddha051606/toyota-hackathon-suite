import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Post-Event Analysis", page_icon="📊", layout="wide")

@st.cache_data
def load_all_lap_data(csv_file_path, lap_reset_threshold_meters=-3000, min_lap_time_seconds=60):
    try:
        # Load data
        cols = ['timestamp', 'telemetry_name', 'telemetry_value']
        df = pd.read_csv(csv_file_path, usecols=cols)
        
        # Pivot
        df_wide = df.pivot_table(index='timestamp', columns='telemetry_name', values='telemetry_value', aggfunc='mean').reset_index()
        
        # Time processing
        df_wide['timestamp_dt'] = pd.to_datetime(df_wide['timestamp'], errors='coerce')
        df_wide = df_wide.sort_values(by='timestamp_dt')
        df_wide['time_sec'] = (df_wide['timestamp_dt'].astype(int) / 10**9)
        
        # Convert columns
        numeric_cols = ['Laptrigger_lapdist_dls', 'speed', 'aps', 'pbrake_f']
        for col in numeric_cols:
            if col in df_wide.columns: 
                df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')
        
        # Fill and Clean
        df_wide = df_wide.ffill().dropna(subset=['time_sec', 'Laptrigger_lapdist_dls'])
        
        # --- OFFICIAL IMS SECTORS (Updated) ---
        # Matches Al Kamel PDF: S1=1364m, S2=2751m, End=Max
        bins = [0.0, 1364.3, 2751.2, 6000.0]
        labels = ["Sector 1", "Sector 2", "Sector 3"]
        df_wide['sector'] = pd.cut(df_wide['Laptrigger_lapdist_dls'], bins=bins, labels=labels, right=True)
        
        # Detect Laps
        df_wide['dist_diff'] = df_wide['Laptrigger_lapdist_dls'].diff()
        lap_crossings = df_wide[df_wide['dist_diff'] < lap_reset_threshold_meters]
        crossing_times = lap_crossings['time_sec'].values
        
        if len(crossing_times) < 2: return None, None
        
        # Extract Laps
        laps_list = []
        for i in range(len(crossing_times) - 1):
            start_t = crossing_times[i]
            end_t = crossing_times[i+1]
            duration = end_t - start_t
            
            if duration > min_lap_time_seconds:
                lap_data = df_wide[(df_wide['time_sec'] >= start_t) & (df_wide['time_sec'] < end_t)].copy()
                
                # Calculate sector times for this lap
                sector_times = lap_data.groupby('sector')['time_sec'].apply(lambda x: x.max() - x.min() if len(x) > 0 else 0.0)
                
                lap_record = {
                    'Lap Number': i + 1,
                    'Lap Time (s)': duration,
                    'S1': sector_times.get("Sector 1", 0),
                    'S2': sector_times.get("Sector 2", 0),
                    'S3': sector_times.get("Sector 3", 0),
                    'data': lap_data 
                }
                laps_list.append(lap_record)
        
        return pd.DataFrame(laps_list), df_wide

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

st.title("📊 Post-Event Analysis")
st.markdown("Deep-dive analysis into race pace, consistency, and potential performance.")

# --- UPDATED PATH FOR INDY ---
csv_path = "indianapolis/R2_indianapolis_motor_speedway_telemetry.csv"
laps_df, full_telemetry = load_all_lap_data(csv_path)

if laps_df is not None:
    # --- Key Statistics ---
    fastest_lap_row = laps_df.loc[laps_df['Lap Time (s)'].idxmin()]
    avg_time = laps_df['Lap Time (s)'].mean()
    
    # --- THEORETICAL BEST LAP ---
    # Now uses the official 3 sectors
    best_s1 = laps_df['S1'].min()
    best_s2 = laps_df['S2'].min()
    best_s3 = laps_df['S3'].min()
    theoretical_best = best_s1 + best_s2 + best_s3
    potential_gain = fastest_lap_row['Lap Time (s)'] - theoretical_best

    st.subheader("Race Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fastest Lap", f"{fastest_lap_row['Lap Time (s)']:.3f}s", f"Lap {fastest_lap_row['Lap Number']}")
    c2.metric("Theoretical Best", f"{theoretical_best:.3f}s", f"-{potential_gain:.3f}s potential")
    c3.metric("Average Lap", f"{avg_time:.3f}s")
    c4.metric("Consistency (StdDev)", f"{laps_df['Lap Time (s)'].std():.3f}s")
    
    st.divider()
    
    # --- THE RACE STORY ---
    st.subheader("📖 The Story of the Race")
    st.markdown(f"""
    * **The Hero Lap:** Lap **{fastest_lap_row['Lap Number']}** was the fastest.
    * **The 'Ideal' Driver:** By combining your best sectors (Official S1, S2, S3), you could have been **{potential_gain:.2f}s faster**.
    * **Consistency Check:** Standard deviation is **{laps_df['Lap Time (s)'].std():.2f}s**.
    """)
    
    st.divider()

    # --- TELEMETRY COMPARISON ---
    st.subheader("🏁 Telemetry Deep Dive")
    st.markdown("Compare your **Fastest Lap** vs. your **Average Lap** to see where speed is lost.")
    
    # Find specific laps to plot
    fast_lap_data = fastest_lap_row['data'].copy()
    avg_lap_idx = (laps_df['Lap Time (s)'] - avg_time).abs().idxmin()
    avg_lap_data = laps_df.loc[avg_lap_idx]['data'].copy()
    
    # Normalize distance 
    fast_lap_data['dist_norm'] = fast_lap_data['Laptrigger_lapdist_dls'] - fast_lap_data['Laptrigger_lapdist_dls'].min()
    avg_lap_data['dist_norm'] = avg_lap_data['Laptrigger_lapdist_dls'] - avg_lap_data['Laptrigger_lapdist_dls'].min()
    
    # Common distance axis
    common_dist = np.linspace(0, 5219, 500) 
    
    fast_speed_interp = np.interp(common_dist, fast_lap_data['dist_norm'], fast_lap_data['speed'])
    avg_speed_interp = np.interp(common_dist, avg_lap_data['dist_norm'], avg_lap_data['speed'])
    
    chart_df = pd.DataFrame({
        'Distance (m)': common_dist,
        'Fastest Lap Speed': fast_speed_interp,
        'Average Lap Speed': avg_speed_interp
    }).set_index('Distance (m)')
    
    st.line_chart(chart_df)
    st.caption("Speed trace comparison. Gaps between lines indicate cornering performance differences.")

    st.divider()
    
    # --- Lap Time Chart ---
    st.subheader("Stint Consistency")
    st.line_chart(laps_df.set_index('Lap Number')['Lap Time (s)'])

else:
    st.info("Processing race data... This may take a moment.")