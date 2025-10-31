import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Pre-Event Prediction",
    page_icon="🔮",
    layout="wide"
)

# --- Helper Functions ---
@st.cache_resource  # Cache the loaded model
def load_model():
    try:
        model = joblib.load("lap_time_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None

def extract_features(lap_df):
    """Extracts the same features we trained on from a lap dataframe."""
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
    return pd.DataFrame([lap_features])

# --- Main Page Layout ---
st.title("🔮 Pre-Event Prediction")
st.markdown("Predict a driver's race pace based on a single practice lap.")

model, model_features = load_model()

if model is None:
    st.error(
        """
        **Model not found!** Please run the `model_trainer.py` script first in your terminal:
        
        `python model_trainer.py` (or your full path)
        """
    )
else:
    st.success("Prediction model loaded successfully.")
    
    # Use our 'ghost_lap.csv' as a stand-in for an "uploaded" practice lap
    st.subheader("Select a Practice Lap to Analyze")
    
    lap_to_predict = st.selectbox(
        "Select a lap:",
        ('Fastest Lap (ghost_lap.csv)', 'Average Lap (live_lap.csv)')
    )
    
    if st.button("Predict Lap Time"):
        if lap_to_predict == 'Fastest Lap (ghost_lap.csv)':
            lap_df = pd.read_csv("ghost_lap.csv")
        else:
            lap_df = pd.read_csv("live_lap.csv")
            
        with st.spinner("Analyzing lap and making prediction..."):
            # 1. Extract Features
            features_df = extract_features(lap_df)
            
            # 2. Re-order features to match model's training
            features_df = features_df.fillna(0) # Handle any NaNs
            features_df = features_df[model_features] # Ensure column order
            
            # 3. Make Prediction
            prediction = model.predict(features_df)
            
            st.subheader("Prediction Result:")
            st.metric(
                label="Predicted Race Pace",
                value=f"{prediction[0]:.3f} seconds"
            )
            st.caption("This is the model's forecast for this driver's average lap time in a race, based on their practice lap inputs.")
            
            st.subheader("Features Extracted from Lap:")
            st.dataframe(features_df)

            # --- NEW: MODEL FEATURE IMPORTANCE ---
            st.subheader("🤖 Model's Reasoning")
            st.markdown("This chart shows which features the model values most when making a prediction.")
            
            # Create a dataframe of features and their importance scores
            importance_df = pd.DataFrame({
                'Feature': model_features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            # Set the feature as the index (better for plotting)
            importance_df = importance_df.set_index('Feature')
            
            # Display the bar chart
            st.bar_chart(importance_df)
            st.caption("A high bar means a change in that feature (e.g., avg_speed) will have a large impact on the predicted lap time.")
            # --- END NEW SECTION ---