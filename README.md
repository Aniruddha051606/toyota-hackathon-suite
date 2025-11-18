# 🏁 Toyota GR Racing Suite
> **"Engineer Victory with Data."**

This is an all-in-one data analytics application built for the **Toyota GR "Hack the Track" Hackathon**. It is a comprehensive suite designed for race engineers and drivers to cover the entire lifecycle of a race event: pre-event strategy, real-time coaching, and post-event analysis.

**Current Configuration:** Indianapolis Motor Speedway (Road Course)

---

## ✨ Features

Our suite is divided into three integrated tools, covering three different hackathon categories in a single application:

### 1. 🔮 Pre-Event Prediction
A forward-looking strategy tool for the **2026 Season**.
* **2026 Strategy Planner:** Select upcoming tracks (e.g., the new **Grand Prix of Arlington**) to simulate race pace using historical data scaling.
* **ML-Powered Forecast:** Uses a `RandomForestRegressor` trained on 2025 telemetry to predict lap times for future events based on driver inputs.
* **Model Reasoning:** A feature importance chart explains *why* the model predicts a specific time (e.g., "Average Speed" vs. "Braking Aggression").

### 2. 🚗 Real-Time Driver Coach
A live simulation dashboard for real-time decision making.
* **G-G Friction Circle:** A professional-grade visualization showing tire usage (lateral vs. longitudinal G-forces) to ensure the driver is at the limit.
* **Live Track Map:** Plots the live driver and 'ghost' car positions on a GPS map in real-time.
* **Corner-by-Corner Analysis:** Automatically detects the track sector and calculates the time delta for that specific section using **Official Al Kamel Systems** timing lines.
* **Actionable Insights:** A smart feedback system that prioritizes instant advice (e.g., *"Ghost is braking, you are not"*) over general sector stats.

### 3. 📊 Post-Event Analysis
A comprehensive report that "tells the story" of the race.
* **The Race Story:** Auto-generates a narrative summary identifying the "Hero Lap," "Problem Lap," and consistency streaks.
* **Theoretical Best Lap:** Calculates the optimal potential lap time by combining the driver's best individual sectors from the entire session.
* **Telemetry Comparison:** Overlays speed traces of the Fastest Lap vs. Average Lap to visually identify cornering deficits.

---

## 🛠️ Tech Stack

* **Core:** Python 3.12+
* **Dashboard:** Streamlit
* **Data Science:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Streamlit Charts
* **Model Storage:** Joblib

---

## 🚀 How to Run This Project

Since the raw telemetry data (3GB) is too large for GitHub, please follow these steps to set up the environment locally.

### 1. Setup Data
1. Download the **`indianapolis.zip`** dataset from the official [Hack the Track website](https://trddev.com/hackathon-2025/).
2. Unzip the file.
3. **Critical Step:** Move the inner `indianapolis` folder into the root of this project directory.
   
   Your folder structure must look like this:
   ```text
   toyota-hackathon-suite/
   ├── indianapolis/
   │   └── R2_indianapolis_motor_speedway_telemetry.csv
   ├── dashboard.py
   ├── ...
   Install Dependencies

### Open your terminal in the project folder and run:

    pip install -r requirements.txt

# 1. Generate the Ghost Lap data (ghost_lap.csv)
    python ghost.py

# 2. Generate the Live Lap data (live_lap.csv)
    python get_live_lap.py

# 3. Train the Prediction Model (lap_time_model.pkl)
    python model_trainer.py
# Launch the Suite 
    streamlit run dashboard.py
