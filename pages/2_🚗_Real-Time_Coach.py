import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Helper Function to Load and Prepare Data ---
@st.cache_data  # This decorator caches the data so it only loads once
def load_and_prepare_data(ghost_path, live_path):
    try:
        ghost_df = pd.read_csv(ghost_path)
        live_df = pd.read_csv(live_path)
        
        ghost_base_time = ghost_df['lap_timestamp'].max()
        
        # --- UPDATED: Added G-Force columns (accx, accy) ---
        telemetry_cols = [
            'lap_timestamp', 'speed', 'nmot', 'aps', 'gear', 
            'Steering_Angle', 'pbrake_f', 'pbrake_r',
            'VBOX_Lat_Min', 'VBOX_Long_Minutes', 'Laptrigger_lapdist_dls',
            'accx_can', 'accy_can' # <-- NEW COLUMNS FOR G-G PLOT
        ]
        
        ghost_df['lap_timestamp'] = pd.to_timedelta(ghost_df['lap_timestamp'], unit='s')
        live_df['lap_timestamp'] = pd.to_timedelta(live_df['lap_timestamp'], unit='s')
        
        ghost_df = ghost_df.set_index('lap_timestamp')
        live_df = live_df.set_index('lap_timestamp')
        
        # Use .reindex() to ensure all expected columns are present
        ghost_df = ghost_df.reindex(columns=telemetry_cols[1:])
        live_df = live_df.reindex(columns=telemetry_cols[1:])
        
        combined_index = pd.to_timedelta(np.arange(
            0, 
            max(ghost_df.index.max().total_seconds(), live_df.index.max().total_seconds()), 
            0.01
        ), unit='s')
        
        ghost_resampled = ghost_df.reindex(ghost_df.index.union(combined_index)).interpolate('time').reindex(combined_index)
        live_resampled = live_df.reindex(live_df.index.union(combined_index)).interpolate('time').reindex(combined_index)
        
        ghost_resampled = ghost_resampled.ffill().bfill()
        live_resampled = live_resampled.ffill().bfill()
        
        live_resampled['delta'] = (live_resampled['speed'] - ghost_resampled['speed']) / 3600 * 0.01
        live_resampled['time_delta_cumulative'] = live_resampled['delta'].cumsum()

        # --- INDIANAPOLIS SECTORS ---
        # --- OFFICIAL IMS SECTORS (Source: Al Kamel Track Map) ---
        # Converted from inches to meters: S1=1364m, S2=2751m, End=Max
        bins = [0.0, 1364.3, 2751.2, 6000.0] 
        labels = ["Sector 1", "Sector 2", "Sector 3"] # IMS only uses 3 sectors
        
        ghost_resampled['sector'] = pd.cut(ghost_resampled['Laptrigger_lapdist_dls'], bins=bins, labels=labels, right=True)
        live_resampled['sector'] = pd.cut(live_resampled['Laptrigger_lapdist_dls'], bins=bins, labels=labels, right=True)
        
        ghost_resampled['sector'] = ghost_resampled['sector'].ffill().bfill()
        live_resampled['sector'] = live_resampled['sector'].ffill().bfill()

        ghost_sector_times = ghost_resampled.groupby('sector').apply(len) * 0.01
        live_sector_times = live_resampled.groupby('sector').apply(len) * 0.01
        
        sector_analysis_df = pd.DataFrame({
            'Ghost Time (s)': ghost_sector_times,
            'Live Time (s)': live_sector_times
        })
        sector_analysis_df['Delta (s)'] = sector_analysis_df['Live Time (s)'] - sector_analysis_df['Ghost Time (s)']
        sector_analysis_df = sector_analysis_df.dropna()

        return ghost_resampled, live_resampled, sector_analysis_df, ghost_base_time
    
    except FileNotFoundError:
        st.error(f"Error: Make sure 'ghost_lap.csv' and 'live_lap.csv' are in the same folder.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None, None, None, None

# --- Main Application ---
st.title("🚗 Real-Time Driver Coach (Indianapolis)")
st.markdown("Comparing the **Fastest Lap** (Ghost) vs. the **Average Lap** (Live Driver)")

ghost, live, sector_analysis, ghost_lap_time = load_and_prepare_data("ghost_lap.csv", "live_lap.csv")

if ghost is not None and live is not None:
    
    total_time_sec = live.index.max().total_seconds()
    
    st.subheader("Race Time Simulator")
    current_time = st.slider(
        "Scrub through the lap (in seconds):", 
        min_value=0.0, 
        max_value=total_time_sec, 
        value=0.0, 
        step=0.1
    )
    
    current_time_td = pd.to_timedelta(current_time, unit='s')
    try:
        ghost_now = ghost.loc[current_time_td]
        live_now = live.loc[current_time_td]
    except KeyError:
        ghost_now = ghost.iloc[ghost.index.get_loc(current_time_td, method='nearest')]
        live_now = live.iloc[live.index.get_loc(current_time_td, method='nearest')]

    time_delta = live_now['time_delta_cumulative']
    
    col_delta, col_projection, col_insight = st.columns([1, 1, 2])
    
    with col_delta:
        st.header(f"Time Delta:")
        st.header(f"{time_delta:+.3f}s")
        if time_delta > 0:
            st.error("🔴 **SLOWER**")
        else:
            st.success("🟢 **FASTER**")

    with col_projection:
        projected_lap_time = ghost_lap_time + time_delta
        st.header("Projected Lap:")
        st.header(f"{projected_lap_time:.3f}s")
        st.caption(f"Ghost Lap Time: {ghost_lap_time:.3f}s")
    
    with col_insight:
        st.subheader("💡 Actionable Insight")
        current_sector = live_now['sector']
        live_brake = 'pbrake_f' in live_now and live_now['pbrake_f'] > 5
        ghost_brake = 'pbrake_f' in ghost_now and ghost_now['pbrake_f'] > 5
        
        if live_brake and not ghost_brake:
            st.warning(f"**Instant Feedback:** You are braking in {current_sector}, but the ghost is not. **Losing time.**")
        elif not live_brake and ghost_brake:
            st.info(f"**Instant Feedback:** The ghost is braking in {current_sector}, but you are not. **Gaining time.**")
        elif live_now['aps'] < 90 and ghost_now['aps'] > 90:
            st.warning(f"**Instant Feedback:** Ghost is full throttle in {current_sector}, but you are not. **Apply more throttle!**")
        elif live_now['speed'] < (ghost_now['speed'] - 5):
            st.warning(f"**Instant Feedback:** Speed is {ghost_now['speed'] - live_now['speed']:.0f} km/h slower than ghost.")
        elif pd.notna(current_sector):
            sector_delta = sector_analysis.loc[current_sector]['Delta (s)']
            if sector_delta > 0.1:
                st.info(f"**Sector Summary:** You lost {sector_delta:.2f}s in {current_sector}.")
            elif sector_delta < -0.1:
                st.info(f"**Sector Summary:** You gained {sector_delta:.2f}s in {current_sector}.")
            else:
                st.success(f"**Sector Summary:** Your pace in {current_sector} matches the ghost.")
        else:
            st.info("Driving...")
            
    st.divider()

    # --- VISUALIZATION ROW (Map + Friction Circle) ---
    col_map, col_gg = st.columns([2, 1])
    
    with col_map:
        st.subheader("🛰️ Live Track Position")
        map_data = pd.DataFrame({
            'lat': [ghost_now['VBOX_Lat_Min'], live_now['VBOX_Lat_Min']],
            'lon': [ghost_now['VBOX_Long_Minutes'], live_now['VBOX_Long_Minutes']]
        })
        map_data = map_data.dropna()
        if not map_data.empty:
            st.map(map_data, zoom=14, use_container_width=True) 
        else:
            st.warning("GPS data not available.")

    # --- NEW: G-G FRICTION CIRCLE ---
    with col_gg:
        st.subheader("🎯 G-G Diagram")
        # Create figure
        fig, ax = plt.subplots(figsize=(4, 4))
        # Draw the "1.5G Limit" circle
        circle = plt.Circle((0, 0), 1.5, color='gray', fill=False, linestyle='--')
        ax.add_artist(circle)
        # Plot the Ghost's entire lap trace (grey background)
        ax.scatter(ghost['accy_can'], ghost['accx_can'], s=1, color='lightgray', alpha=0.3)
        # Plot the Live Driver's CURRENT position (red dot)
        ax.scatter(live_now['accy_can'], live_now['accx_can'], s=100, color='red', edgecolors='black', label='Live')
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel("Lateral G (Turning)")
        ax.set_ylabel("Longitudinal G (Accel/Brake)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')
        st.pyplot(fig)
    # --- END NEW SECTION ---

    st.divider()
    
    st.subheader("Live Telemetry Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👻 Ghost (Fastest Lap)")
        g_col1, g_col2 = st.columns(2)
        g_col1.metric("Speed (km/h)", f"{ghost_now['speed']:.1f}")
        g_col2.metric("RPM", f"{ghost_now['nmot']:.0f}")
        g_col1.metric("Steering", f"{ghost_now['Steering_Angle']:.1f}°")
        g_col2.metric("Gear", f"{ghost_now['gear']:.0f}")
        
        st.text("Throttle:")
        st.progress(int(ghost_now['aps']))
        st.text("Brake (Front):")
        st.progress(int(ghost_now['pbrake_f']) if 'pbrake_f' in ghost_now and pd.notna(ghost_now['pbrake_f']) else 0)


    with col2:
        st.subheader("🚗 Live Driver (Average Lap)")
        l_col1, l_col2 = st.columns(2)
        l_col1.metric("Speed (km/h)", f"{live_now['speed']:.1f}")
        l_col2.metric("RPM", f"{live_now['nmot']:.0f}")
        l_col1.metric("Steering", f"{live_now['Steering_Angle']:.1f}°")
        l_col2.metric("Gear", f"{live_now['gear']:.0f}")

        st.text("Throttle:")
        st.progress(int(live_now['aps']))
        st.text("Brake (Front):")
        st.progress(int(live_now['pbrake_f']) if 'pbrake_f' in live_now and pd.notna(live_now['pbrake_f']) else 0)
        
    st.divider()
    
    st.subheader("Full Lap Analysis")

    st.markdown("This chart shows the total time lost or gained in each sector. (Negative = Faster)")
    st.bar_chart(sector_analysis['Delta (s)'])

    speed_chart_df = pd.DataFrame({
        'Ghost Lap': ghost['speed'],
        'Live Lap': live['speed']
    })
    speed_chart_df.index = speed_chart_df.index.total_seconds()
    speed_chart_df.index.name = "Time (s)"
    
    st.line_chart(speed_chart_df, use_container_width=True)
    st.caption("Speed (km/h) over the full lap (in seconds).")

else:
    st.info("Waiting for data files... (Make sure 'ghost_lap.csv' and 'live_lap.csv' are in the same folder)")