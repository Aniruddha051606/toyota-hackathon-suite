import streamlit as st

st.set_page_config(
    page_title="Toyota GR Racing Suite",
    page_icon="🏁",
    layout="wide"
)

st.title("Welcome to the Toyota GR Racing Suite 🏁")
st.subheader("Your all-in-one tool for race analysis.")
st.markdown("---")
st.markdown(
    """
    This application is a proof-of-concept for the 'Hack the Track' hackathon,
    combining three analysis modes into one powerful suite:
    
    * **🔮 Pre-Event Prediction:** (Select from the sidebar)
        A machine learning model to forecast race pace based on practice data.
        
    * **🚗 Real-Time Coach:** (Select from the sidebar)
        A live dashboard comparing a driver's inputs against a 'ghost' lap,
        providing actionable feedback and a predictive lap timer.
        
    * **📊 Post-Event Analysis:** (Select from the sidebar)
        A full-race breakdown showing lap-by-lap consistency, key statistics,
        and a narrative "race story".
    
    **To get started, please select a tool from the sidebar on the left.**
    """
)