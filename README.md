# Sentinel Dispatch: Predictive Emergency Response Turnout
<img width="3018" height="1596" alt="image" src="https://github.com/user-attachments/assets/b216561b-7a0b-4d34-ab24-817376f765b9" />


## Project Overview
**Sentinel Dispatch Project Overview** 
Sentinel Dispatch is a Deep Learning and Agentic AI project designed to predict emergency response turnout times (the interval between dispatch and unit movement) and provide real-time tactical advisories.

By utilizing high-velocity, structured synthetic data, this project establishes a predictive model and an automated dispatch reasoning layer. The system is designed to demonstrate how high-fidelity data and LPU-accelerated inference can optimize critical infrastructure in major metropolitan areas like Seattle or London.

While many developing regions still rely on analog record-keeping, this project utilizes high-velocity, structured synthetic data to establish a **"Gold Standard" predictive model**.

## Phase 1: Synthetic Data Generation

Developed an advanced Python generator that simulates 5,000 incidents with realistic correlations:

Spatial Mapping: Integrated real-world Seattle Fire Department assets (Stations 10, 18, 22, 28) and neighborhood-specific traffic multipliers.

Dynamic Hazards: Simulated road closures, weather indices (storm/snow), and peak rush hour patterns.

Geospatial Logic: Implemented Haversine distance calculations to establish proximity-based response delays.

Seattle Fire Station 10: https://www.google.com/maps?cid=17687369618003697318&g_mp=Cidnb29nbGUubWFwcy5wbGFjZXMudjEuUGxhY2VzLlNlYXJjaFRleHQ 
Seattle Fire Station 18: https://www.google.com/maps?cid=8926784321114591044&g_mp=Cidnb29nbGUubWFwcy5wbGFjZXMudjEuUGxhY2VzLlNlYXJjaFRleHQ
Seattle Fire Station 22: https://www.google.com/maps?cid=15065013962951630969&g_mp=Cidnb29nbGUubWFwcy5wbGFjZXMudjEuUGxhY2VzLlNlYXJjaFRleHQ
and Seattle Fire Station 28: https://www.google.com/maps?cid=12092757799789534514&g_mp=Cidnb29nbGUubWFwcy5wbGFjZXMudjEuUGxhY2VzLlNlYXJjaFRleHQ

Dynamic Hazards: Simulated road closures, weather indices (storm/snow), and neighborhood-specific traffic multipliers.

Geospatial Logic: Implemented Haversine distance calculations to establish proximity-based response delays.
<img width="1222" height="732" alt="image" src="https://github.com/user-attachments/assets/5a856922-2e10-4606-8e89-6800c5854b23" />


## Phase 2 & 3: Preprocessing & Deep Learning
Engineering: Applied One-Hot Encoding for categorical variables and Cyclical Feature Engineering (Sin/Cos transforms) for temporal data.

Architecture: Built a 4-layer Fully Connected Neural Network (Dense) using TensorFlow/Keras.

Results: Achieved a Mean Absolute Error (MAE) of 9.21 seconds, ensuring high predictive precision across diverse environmental and logistical scenarios.

## Phase 4 & 5: Agentic Orchestration & LPU Inference
Implemented a modular AI Agent using LangChain Expression Language (LCEL) to provide natural language advisories:

Inference Engine: Powered by Groq LPU (Language Processing Unit) for sub-second response times.

Model: Utilizes Meta Llama 3.1 (8B-Instant) to synthesize predictive outputs with spatial hazard data.

Functional Output: The agent evaluates predicted turnout times against active road closures to provide tactical rerouting recommendations.

## Sentinel Predictor (Inference)
The project includes a functional inference pipeline. For example, a Priority 1 Medical call during a storm at 5:00 PM yields:
**`Predicted Turnout: 93.17 seconds`**

##Visual Analytics
The project includes an interactive geospatial heatmap (built with Folium) that identifies "logistical dark spots" and visualizes regional bottlenecks across the Seattle urban grid.

## Model Performance
<img width="584" height="382" alt="Sentinel Learning Curve" src="https://github.com/user-attachments/assets/be215eef-9855-42c1-acb4-78ae93d7b2e7" />

## Tech Stack
- AI/ML: TensorFlow, Keras, Scikit-Learn.

- Orchestration: LangChain (LCEL), Llama 3.1, Groq LPU API.

- Data Engineering: Python, Pandas, NumPy.

- Geospatial: Folium (Heatmaps), Geopy/Haversine.

- Environment: Google Colab / VS Code.

## Future Roadmap
Future Roadmap
Real-time API: Deployment via FastAPI for external integration.
Multi-Agent Systems: Implementing LangGraph for complex resource allocation between multiple fire and medical units.
