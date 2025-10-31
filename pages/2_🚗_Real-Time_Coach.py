import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.io.wavfile import write as write_wav

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

# --- NEW: AUDIO GENERATION FUNCTION ---
@st.cache_data # Cache the audio file once it's generated
def generate_lap_audio(lap_csv_path):
    """
    Generates an audio representation of a lap using RPM for pitch
    and Throttle for volume. This is 'Telemetry Sonification'.
    """
    try:
        df = pd.read_csv(lap_csv_path)
        
        # Prepare RPM data (for pitch)
        rpm = df['nmot'].fillna(2000) # Fill gaps with idle RPM
        min_rpm, max_rpm = 2000, 8000 # Guesses for engine range
        min_freq, max_freq = 150, 600 # Pitch range (in Hz)
        # Normalize RPM to 0-1 range
        rpm_normalized = (rpm - min_rpm) / (max_rpm - min_rpm)
        # Map to frequency range
        frequency = (rpm_normalized * (max_freq - min_freq)) + min_freq
        
        # Prepare Throttle data (for volume)
        throttle = df['aps'].fillna(0) / 100.0 # Normalize 0-100 to 0-1
        
        # Prepare time data
        sample_rate = 44100
        time = df['lap_timestamp']
        duration = time.max()
        
        # Create an evenly-spaced time array for the audio signal
        t_audio = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
        
        # Interpolate our data to match the new audio time array
        frequency_interp = np.interp(t_audio, time, frequency)
        throttle_interp = np.interp(t_audio, time, throttle)

        # Generate the audio wave
        # Create the phase by integrating frequency (rpm)
        phase = np.cumsum(2 * np.pi * frequency_interp / sample_rate)
        # Create the signal by multiplying the wave by amplitude (throttle)
        signal = (throttle_interp * np.sin(phase)).astype(np.float32)
        
        # Normalize to 16-bit audio range
        signal_normalized = np.int16(signal / np.max(np.abs(signal)) * 32767)
        
        # Save to an in-memory buffer
        buffer = io.BytesIO()
        write_wav(buffer, sample_rate, signal_normalized)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Could not generate audio: {e}")
        return None
# --- END NEW FUNCTION ---


# --- Main Page Layout ---
st.title("📊 Post-Event Analysis")
st.markdown("Analyzing every lap from the race to find key moments and driver consistency.")

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
    
    # --- NEW: THE RACE STORY ---
    st.subheader("📖 The Story of the Race")
    
    st.markdown(f"* **The Hero Lap:** Lap **{fastest_lap_num}** was the fastest at **{fastest_lap_time:.3f}s**.")
    st.markdown(f"* **The Problem Lap:** Lap **{slowest_lap_num}** was the slowest at **{slowest_lap_time:.3f}s**, a **{slowest_lap_time - fastest_lap_time:.3f}s** difference.")
    
    # Analyze a mid-race stint
    stint_start, stint_end = 5, 10
    stint_laps = all_laps[(all_laps['Lap Number'] >= stint_start) & (all_laps['Lap Number'] <= stint_end)]
    if not stint_laps.empty:
        stint_std = stint_laps['Lap Time (s)'].std()
        st.markdown(f"* **Consistent Stint:** Laps {stint_start}-{stint_end} were highly consistent, with times varying by only {stint_std:.3f}s (Std. Dev).")
    
    # Analyze tire degradation at the end
    final_laps = all_laps.tail(5)
    if len(final_laps) > 1:
        degradation = final_laps['Lap Time (s)'].diff().mean()
        if degradation > 0.1:
            st.markdown(f"* **Tire Wear:** In the final {len(final_laps)} laps, a clear degradation pattern emerged, with times increasing by an average of **{degradation:.3f}s** per lap.")
    # --- END NEW SECTION ---

    st.divider()
    
    # --- Lap Time Chart ---
    st.subheader("Lap Time Consistency")
    st.markdown("This chart shows your lap time over the entire race. Look for a stable, flat line.")
    chart_data = all_laps.set_index('Lap Number')
    st.line_chart(chart_data['Lap Time (s)'])

    st.divider()

    
    st.subheader("🎧 Wildcard: Listen to the Lap")
    st.markdown("Hear the 'song' of the ghost lap. The **pitch is the engine RPM**, and the **volume is the throttle**.")
    
    if st.button("Generate Audio for Ghost Lap"):
        with st.spinner("Sonifying telemetry data... This may take a few moments."):
            audio_bytes = generate_lap_audio("ghost_lap.csv")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
    # --- END NEW SECTION ---

else:
    st.info("Processing race data... This may take a moment.")