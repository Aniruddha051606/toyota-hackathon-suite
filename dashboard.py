import streamlit as st

# --- Page Config ---
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Toyota GR Racing Suite",
    page_icon="🏁",
    layout="wide"
)

# --- Main Page Content ---
st.title("Welcome to the Toyota GR Racing Suite 🏁")
st.subheader("Your all-in-one tool for race analysis.")
st.markdown("---")
st.markdown(
    """
    This application is a proof-of-concept for the 'Hack the Track' hackathon,
    combining three analysis modes into one powerful suite:
    
    * **🚗 Real-Time Coach:** (Select from the sidebar)
        A live dashboard comparing a driver's inputs against a 'ghost' lap,
        providing actionable feedback and a predictive lap timer.
        
    * **📊 Post-Event Analysis:** (Select from the sidebar)
        A full-race breakdown showing lap-by-lap consistency, key statistics,
        and pace degradation.
    
    * **🔮 Pre-Event Prediction:** (Coming Soon)
        A machine learning model to forecast race pace based on practice data.
    
    **To get started, please select a tool from the sidebar on the left.**
    """
)