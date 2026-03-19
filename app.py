
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# 1. Page Configuration
st.set_page_config(page_title="Sentinel Dispatch AI", layout="wide")
st.title("🚨 Sentinel Dispatch: Emergency Logistics Dashboard")

# 2. Sidebar for Inputs
st.sidebar.header("Incident Configuration")
# We use the key names we used in our data generator
zip_code = st.sidebar.selectbox("Neighborhood Zip Code", ["98101", "98107", "98102", "98118"])
priority = st.sidebar.slider("Priority Level (1=Highest)", 1, 3, 1)
incident_type = st.sidebar.selectbox("Incident Type", ["Medical", "Fire", "Traffic", "Structure"])
weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Rain", "Storm", "Snow"])

# 3. Load Assets
@st.cache_data
def load_assets():
    return [
        {"name": "Station 10 (Downtown)", "loc": [47.6011, -122.3285]},
        {"name": "Station 18 (Ballard)", "loc": [47.6683, -122.3772]},
        {"name": "Station 22 (Capitol Hill)", "loc": [47.6429, -122.3209]},
        {"name": "Station 28 (Rainier)", "loc": [47.5487, -122.2765]}
    ]

# 4. Agent Logic (Simplified for the Dashboard)
def get_agent_response(z, p, i, w):
    # Using st.secrets for when we deploy to Streamlit Cloud later
    # For now, we'll try to get it from environment variables
    api_key = os.environ.get("GROQ_API_KEY") 
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)
    
    # Static prediction for dashboard demo
    prediction = 48.5 if p == 1 else 102.3
    hazards = 15 if z == "98101" else 0
    
    prompt = ChatPromptTemplate.from_template("Provide a 2-sentence dispatch advisory for {i} in {z}. Turnout: {t}s, Hazards: {h}.")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"i": i, "z": z, "t": prediction, "h": hazards})

# 5. Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Geospatial Hazard Map")
    m = folium.Map(location=[47.6062, -122.3321], zoom_start=11)
    for s in load_assets():
        folium.Marker(s['loc'], popup=s['name'], icon=folium.Icon(color='red', icon='fire')).add_to(m)
    st_folium(m, width=700, height=500)

with col2:
    st.subheader("Sentinel AI Advisory")
    if st.button("Generate Dispatch Report"):
        with st.spinner("Analyzing..."):
            report = get_agent_response(zip_code, priority, incident_type, weather)
            st.info(report)
            st.metric("Predicted Turnout", "48.5s", "-4.2s")
