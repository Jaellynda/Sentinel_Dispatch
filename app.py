
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

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("🚨 Incident Parameters")
zip_code = st.sidebar.selectbox("Zone (Zip)", ["98101", "98107", "98102", "98118"])
priority = st.sidebar.select_slider("Priority", options=[3, 2, 1])
incident = st.sidebar.selectbox("Type", ["Medical", "Fire", "Traffic", "Structure"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Storm", "Snow"])

# --- 3. MOCKED ENGINE LOGIC (For Portfolio Demo) ---
# Note: In a full production app, you would load your .h5 model and scaler here.
def get_prediction_and_shap(z, p, i, w):
    # Simulated DNN prediction logic
    base_time = 90
    p_mod = -30 if p == 1 else 10
    w_mod = 25 if w in ["Storm", "Snow"] else 0
    final_pred = base_time + p_mod + w_mod
    
    # Return simulated SHAP values for the force plot
    # Values represent: [Priority, Weather, Zone, Type]
    return final_pred, np.array([p_mod, w_mod, 5, 2])

# --- 4. LAYOUT: MAP & AGENT ---
st.title("Sentinel Dispatch: Autonomous Logistics & XAI")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Geospatial Operational View")
    m = folium.Map(location=[47.6062, -122.3321], zoom_start=12, tiles="cartodbpositron")
    stations = [
        {"n": "Station 10", "l": [47.6011, -122.3285]},
        {"n": "Station 18", "l": [47.6683, -122.3772]}
    ]
    for s in stations:
        folium.Marker(s['l'], popup=s['n'], icon=folium.Icon(color='red', icon='fire')).add_to(m)
    st_folium(m, width="100%", height=450)

with col2:
    st.subheader("Sentinel AI Advisory")
    if st.button("Run Dispatch Analytics"):
        pred, shap_vals = get_prediction_and_shap(zip_code, priority, incident, weather)
        
        # Groq Agent Logic
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if api_key:
            llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)
            prompt = ChatPromptTemplate.from_template("Explain a {pred}s turnout for a {i} call in {w} weather. Be brief.")
            chain = prompt | llm | StrOutputParser()
            st.info(chain.invoke({"pred": pred, "i": incident, "w": weather}))
        
        st.metric("Predicted Turnout", f"{pred} Seconds", delta="-4.2s" if priority==1 else "Delay Expected")

# --- 5. EXPLAINABLE AI (XAI) SECTION ---
st.divider()
st.subheader("Decision Transparency (SHAP Explainer)")
st.write("This section visualizes the 'Why' behind the prediction using Shapley values.")

# Force Plot logic
# We simulate a force plot to show recruiters how the model weights features
expected_value = 85
if st.checkbox("Show Model Logic"):
    # Using matplotlib=False to get the interactive HTML version
    p = shap.force_plot(
        expected_value, 
        np.array([-20, 15, 5, -2]), # Mocked SHAP values
        feature_names=["Priority", "Weather", "Zone", "Type"],
        matplotlib=False
    )
    st_shap(p, height=150)
