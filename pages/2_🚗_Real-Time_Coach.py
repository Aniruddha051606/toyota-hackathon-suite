import streamlit as st
import pandas as pd
import numpy as np

# --- Helper Function to Load and Prepare Data ---
@st.cache_data  # This decorator caches the data so it only loads once
def load_and_prepare_data(ghost_path, live_path):
    try:
        ghost_df = pd.read_csv(ghost_path)
        live_df = pd.read_csv(live_path)
        
        ghost_base_time = ghost_df['lap_timestamp'].max()
        
        telemetry_cols = [
            'lap_timestamp', 'speed', 'nmot', 'aps', 'gear', 
            'Steering_Angle', 'pbrake_f', 'pbrake_r',
            'VBOX_Lat_Min', 'VBOX_Long_Minutes', 'Laptrigger_lapdist_dls'
        ]
        
        ghost_df['lap_timestamp'] = pd.to_timedelta(ghost_df['lap_timestamp'], unit='s')
        live_df['lap_timestamp'] = pd.to_timedelta(live_df['lap_timestamp'], unit='s')
        
        ghost_df = ghost_df.set_index('lap_timestamp')
        live_df = live_df.set_index('lap_timestamp')
        
        ghost_df = ghost_df.reindex(columns=telemetry_cols[1:])
        live_df = live_df.reindex(columns=telemetry_cols[1:])
        
        rule = '0.01S' # 10ms resampling
        
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

        # --- NEW: SECTOR-BY-SECTOR ANALYSIS ---
        # Use the values from your inspector script
        bins = [4.0, 1216.0, 1608.0, 1943.0, 3699.0]
        labels = ["Sector 1", "Sector 2", "Sector 3", "Sector 4"]
        
        # Assign each row to a sector
        ghost_resampled['sector'] = pd.cut(ghost_resampled['Laptrigger_lapdist_dls'], bins=bins, labels=labels, right=True)
        live_resampled['sector'] = pd.cut(live_resampled['Laptrigger_lapdist_dls'], bins=bins, labels=labels, right=True)
        
        # Fill any gaps (e.g., at the start)
        ghost_resampled['sector'] = ghost_resampled['sector'].ffill().bfill()
        live_resampled['sector'] = live_resampled['sector'].ffill().bfill()

        # Calculate time spent in each sector (each row is 0.01s)
        ghost_sector_times = ghost_resampled.groupby('sector').apply(len) * 0.01
        live_sector_times = live_resampled.groupby('sector').apply(len) * 0.01
        
        # Create the analysis DataFrame
        sector_analysis_df = pd.DataFrame({
            'Ghost Time (s)': ghost_sector_times,
            'Live Time (s)': live_sector_times
        })
        sector_analysis_df['Delta (s)'] = sector_analysis_df['Live Time (s)'] - sector_analysis_df['Ghost Time (s)']
        sector_analysis_df = sector_analysis_df.dropna()
        # --- END NEW SECTOR ANALYSIS ---

        return ghost_resampled, live_resampled, sector_analysis_df, ghost_base_time
    
    except FileNotFoundError:
        st.error(f"Error: Make sure 'ghost_lap.csv' and 'live_lap.csv' are in the same folder.")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None, None, None, None


# --- Main Application ---
st.title("🚗 Real-Time Driver Coach")
st.markdown("Comparing the **Fastest Lap** (Ghost) vs. the **Average Lap** (Live Driver)")

# Load the data
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
        # Fallback for slider precision
        ghost_now = ghost.iloc[ghost.index.get_loc(current_time_td, method='nearest')]
        live_now = live.iloc[live.index.get_loc(current_time_td, method='nearest')]

    time_delta = live_now['time_delta_cumulative']
    
    # --- UPGRADED TOP SECTION ---
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
    
    # --- UPGRADED: ACTIONABLE INSIGHT (SECTOR-AWARE) ---
    with col_insight:
        st.subheader("💡 Actionable Insight")
        
        # Get current sector
        current_sector = live_now['sector']
        
        if pd.isna(current_sector):
            st.info("Driving...")
        else:
            sector_delta = sector_analysis.loc[current_sector]['Delta (s)']
            
            # High-level sector insight
            if sector_delta > 0.1:
                st.warning(f"**Losing {sector_delta:.2f}s in {current_sector}.**")
            elif sector_delta < -0.1:
                st.success(f"**Gaining {sector_delta:.2f}s in {current_sector}!**")
            else:
                st.info(f"Pace is matching the ghost in {current_sector}.")

            # Low-level instant insight
            live_brake = 'pbrake_f' in live_now and live_now['pbrake_f'] > 5
            ghost_brake = 'pbrake_f' in ghost_now and ghost_now['pbrake_f'] > 5
            
            if live_brake and not ghost_brake:
                st.markdown("*Instant Feedback:* You are braking, but the ghost is not.")
            elif not live_brake and ghost_brake:
                st.markdown("*Instant Feedback:* Ghost is braking, but you are not.")
            elif live_now['aps'] < 90 and ghost_now['aps'] > 90:
                st.markdown("*Instant Feedback:* Ghost is full throttle, but you are not.")
            elif live_now['speed'] < (ghost_now['speed'] - 5):
                st.markdown(f"*Instant Feedback:* Speed is {ghost_now['speed'] - live_now['speed']:.0f} km/h slower.")
    # --- END UPGRADED INSIGHT ---
            
    st.divider()

    # --- LIVE TRACK MAP ---
    st.subheader("🛰️ Live Track Position")
    map_data = pd.DataFrame({
        'lat': [ghost_now['VBOX_Lat_Min'], live_now['VBOX_Lat_Min']],
        'lon': [ghost_now['VBOX_Long_Minutes'], live_now['VBOX_Long_Minutes']]
    })
    map_data = map_data.dropna()
    if not map_data.empty:
        st.map(map_data, zoom=15, use_container_width=True) 
    else:
        st.warning("GPS data not available for this timestamp.")

    st.divider()
    
    # --- LIVE TELEMETRY WITH VISUALS ---
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

    # --- NEW: SECTOR DELTA BAR CHART ---
    st.markdown("This chart shows the total time lost or gained in each sector. (Negative = Faster)")
    st.bar_chart(sector_analysis['Delta (s)'])
    # --- END NEW CHART ---

    # --- Full Lap Line Charts ---
    speed_chart_df = pd.DataFrame({
        'Ghost Lap': ghost['speed'],
        'Live Lap': live['speed']
    })
    speed_chart_df.index = speed_chart_df.index.total_seconds()
    speed_chart_df.index.name = "Time (s)"
    
    st.line_chart(speed_chart_df, use_container_width=True)
    st.caption("Speed (km/h) over the full lap (in seconds).")

    throttle_chart_df = pd.DataFrame({
        'Ghost Lap': ghost['aps'],
        'Live Lap': live.get('aps', pd.Series(index=live.index, name='aps'))
    })
    throttle_chart_df.index = throttle_chart_df.index.total_seconds()
    throttle_chart_df.index.name = "Time (s)"
    
    st.line_chart(throttle_chart_df, use_container_width=True)
    st.caption("Throttle Application (%) over the full lap (in seconds).")

else:
    st.info("Waiting for data files... (Make sure 'ghost_lap.csv' and 'live_lap.csv' are in the same folder)")
