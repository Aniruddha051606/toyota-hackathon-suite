# 🏁 GR Racing Suite (A "Hack the Track" Project)

This is an all-in-one data analytics application for the **Toyota GR "Hack the Track" Hackathon**. It's a multi-page suite designed for race engineers and drivers to cover the entire lifecycle of a race event: pre-event strategy, real-time coaching, and post-event analysis.

This project is built as a single, powerful **Streamlit** application.

---

## ✨ Features (The "Good Attributes")

Our suite is divided into three pages, covering three different hackathon categories:

### 1. 🔮 Pre-Event Prediction
This page uses a Machine Learning model to forecast a driver's race pace based on a single practice lap.

* **ML-Powered Forecast:** Uses a `RandomForestRegressor` model trained on all telemetry data to predict a driver's average lap time.
* **Model Reasoning:** Includes a "Feature Importance" chart to show *why* the model made its prediction (e.g., "avg_speed" and "percent_full_throttle" were the most important factors).

### 2. 🚗 Real-Time Driver Coach
This is a live analysis dashboard that simulates a driver's lap and compares it against the "ghost" (fastest possible) lap.

* **Live Track Map:** A "wow" feature that plots the live driver and the ghost car's positions on a map in real-time using GPS data.
* **Corner-by-Corner Analysis:** The track is broken into 4 sectors. The app calculates the time delta *per sector* and shows exactly where the driver is losing time.
* **Predictive Lap Timer:** A real-time counter that forecasts the driver's final lap time based on their current performance.
* **Actionable Insights:** A dynamic text box gives the driver plain-English advice (e.g., "**Losing 0.15s in Sector 3:** You are braking 10m earlier than the ghost.").
* **Visual Telemetry:** Clean, side-by-side progress bars show the throttle and brake inputs for both drivers, making comparison instant.

### 3. 📊 Post-Event Analysis
A comprehensive report that "tells the story" of a race, going beyond simple stats.

* **The Race Story:** Auto-generates a narrative summary of the race, identifying the "Hero Lap" (fastest), the "Problem Lap" (slowest), and periods of high consistency.
* **Wildcard Feature: Telemetry Sonification:** A unique button that lets you **"Listen to the Lap."** It converts the ghost lap's telemetry into an audio file, where engine **RPM controls the pitch** and **throttle controls the volume**.
* **Full-Race Consistency:** A line chart that plots every lap time from the race, making it easy to spot driver fatigue or tire degradation.

---

## 🛠️ Tech Stack

* **Core:** Python
* **Dashboard:** Streamlit
* **Data Manipulation:** Pandas & NumPy
* **Machine Learning:** Scikit-learn (RandomForestRegressor)
* **Audio Generation:** SciPy
* **Model Storage:** Joblib

---

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas scikit-learn scipy
    ```

2.  **Train the ML Model:**
    (You only need to do this once. It will create the `lap_time_model.pkl` file.)
    ```bash
    python model_trainer.py
    ```

3.  **Run the App:**
    ```bash
    streamlit run dashboard.py
    ```