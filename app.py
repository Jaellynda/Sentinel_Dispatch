import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
import shap
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Sentinel Dispatch AI", layout="wide")

# --- 2. HELPER FUNCTIONS (Defined before they are called) ---
@st.cache_data
def load_assets():
    """Returns the list of emergency stations for the map"""
    return [
        {"name": "Station 10 (Downtown)", "loc": [47.6011, -122.3285]},
        {"name": "Station 18 (Ballard)", "loc": [47.6683, -122.3772]},
        {"name": "Station 22 (Capitol Hill)", "loc": [47.6429, -122.3209]},
        {"name": "Station 28 (Rainier)", "loc": [47.5487, -122.2765]}
    ]

def st_shap(plot, height=None):
    """Renders SHAP interactive plots in Streamlit"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def get_prediction_and_shap(z, p, i, w):
    """Simulated DNN logic for the demo"""
    base_time = 90
    p_mod = -30 if p == 1 else 10
    w_mod = 25 if w in ["Storm", "Snow"] else 0
    final_pred = base_time + p_mod + w_mod
    # Mock SHAP values [Priority, Weather, Zone, Type]
    return final_pred, np.array([p_mod, w_mod, 5, 2])

# --- 3. SIDEBAR CONFIGURATION ---
st.sidebar.header("🚨 Incident Parameters")
zip_code = st.sidebar.selectbox("Zone (Zip)", ["98101", "98107", "98102", "98118"])
priority = st.sidebar.select_slider("Priority", options=[3, 2, 1])
incident = st.sidebar.selectbox("Type", ["Medical", "Fire", "Traffic", "Structure"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Storm", "Snow"])

# --- 4. LAYOUT: MAP & AGENT ---
st.title("Sentinel Dispatch: Autonomous Logistics & XAI")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Geospatial Operational View")
    # Initialize Map centered on Seattle
    m = folium.Map(location=[47.6062, -122.3321], zoom_start=11, tiles="cartodbpositron")
    
    # Dynamically plot all stations from the helper function
    all_stations = load_assets() 
    for s in all_stations:
        folium.Marker(
            location=s['loc'], 
            popup=s['name'], 
            icon=folium.Icon(color='red', icon='fire')
        ).add_to(m)
        
    st_folium(m, width="100%", height=450)

with col2:
    st.subheader("Sentinel AI Advisory")
    if st.button("Run Dispatch Analytics"):
        pred, shap_vals = get_prediction_and_shap(zip_code, priority, incident, weather)
        
        # Groq Agent Logic
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)
                prompt = ChatPromptTemplate.from_template("Explain a {pred}s turnout for a {i} call in {w} weather. Be brief.")
                chain = prompt | llm | StrOutputParser()
                st.info(chain.invoke({"pred": pred, "i": incident, "w": weather}))
            except Exception as e:
                st.warning("Agent is currently offline. Viewing core metrics only.")
        
        # Metric display with consistency check for 'priority'
        st.metric("Predicted Turnout", f"{pred} Seconds", delta="-4.2s" if priority == 1 else "Delay Expected")

# --- 5. EXPLAINABLE AI (XAI) SECTION ---
st.divider()
st.subheader("Decision Transparency (SHAP Explainer)")
st.write("This section visualizes the 'Why' behind the prediction using Shapley values.")

if st.checkbox("Show Model Logic"):
    expected_value = 85
    # Render interactive Force Plot
    p_plot = shap.force_plot(
        expected_value, 
        np.array([-20, 15, 5, -2]), 
        feature_names=["Priority", "Weather", "Zone", "Type"],
        matplotlib=False
    )
    st_shap(p_plot, height=150)
