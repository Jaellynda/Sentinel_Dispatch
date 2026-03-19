# Sentinel Dispatch: Predictive Emergency Response Turnout

## Project Overview
**Sentinel Dispatch** is a Deep Learning project designed to predict emergency response "turnout times" (the interval between dispatch and unit movement). 

While many developing regions still rely on analog record-keeping, this project utilizes high-velocity, structured synthetic data to establish a **"Gold Standard" predictive model**. The goal is to demonstrate how high-fidelity data can be used to optimize dispatch systems in major metropolitan areas like Seattle or London.

## Phase 1: Synthetic Data Generation
Developed a custom Python generator (`data_generator.py`) to simulate 5,000 incidents with realistic correlations:
- **Priority Logic:** Priority 1 calls show significantly faster turnout times.
- **Environmental Impact:** Integrated delays for storms, snow, and peak rush hour (7-9 AM, 4-6 PM).
- **Apparatus Prep:** Accounted for the slower "spin-up" time of heavy engines vs. light rescue units.

## Phase 2 & 3: Preprocessing & Deep Learning
- **Engineering:** Applied **One-Hot Encoding** for categorical variables and **Cyclical Feature Engineering** (Sin/Cos transforms) for time-of-day data to ensure the model understands temporal loops.
- **Architecture:** Built a 4-layer Fully Connected Neural Network (Dense) using **TensorFlow/Keras**.
- **Results:** Achieved a **Mean Absolute Error (MAE) of 9.21 seconds**, indicating high predictive precision across diverse environmental scenarios.

## Phase 4: Sentinel Predictor (Inference)
The project includes a functional inference pipeline. For example, a Priority 1 Medical call during a storm at 5:00 PM yields:
**`Predicted Turnout: 93.17 seconds`**

## Model Performance
<img width="584" height="382" alt="Sentinel Learning Curve" src="https://github.com/user-attachments/assets/be215eef-9855-42c1-acb4-78ae93d7b2e7" />

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn (Preprocessing), TensorFlow/Keras (Modeling), Matplotlib (Visualization)
- **Environment:** Google Colab / VS Code

## Future Roadmap
- **Deployment:** Wrap the model in a FastAPI or Flask app.
- **Optimization:** Implement Groq LPU inference for sub-millisecond dispatch predictions.
