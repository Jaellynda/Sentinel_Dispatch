
# Sentinel Dispatch: Predictive Emergency Response Turnout

## Project Overview
**Sentinel Dispatch** is a Deep Learning project designed to predict emergency response "turnout times" (the interval between dispatch and unit movement). 

While many developing regions still rely on analog record-keeping, this project utilizes high-velocity, structured synthetic data to establish a **"Gold Standard" predictive model**. The goal is to demonstrate how high-fidelity data can be used to optimize dispatch systems in major metropolitan areas like Seattle or London.

## Phase 1: Synthetic Data Generation
To ensure the model learns realistic operational patterns, I developed a custom Python generator (`data_generator.py`) that simulates 5,000 emergency incidents with the following features:
- **Incident Type:** (Medical, Fire, Traffic, etc.)
- **Priority Level:** Weighted impact on turnout speed.
- **Environmental Factors:** Weather Index (Storm/Snow delays) and Rush Hour patterns.
- **Unit Type:** Accounting for the physical prep time of different apparatus (Engines vs. Ambulances).

## Tech Stack
- **Language:** Python
- **Environment:** Google Colab / VS Code
- **Libraries:** Pandas, NumPy (Data Engineering); TensorFlow/Keras (Upcoming DNN)
- **Version Control:** Git/GitHub

## Next Steps
- **Phase 2:** Feature Engineering & Preprocessing (One-Hot Encoding for categorical data).
- **Phase 3:** Building and training a Deep Neural Network (DNN) to minimize Mean Absolute Error (MAE) in turnout predictions.

-<img width="584" height="382" alt="Screenshot 2026-03-17 at 9 12 16 PM" src="https://github.com/user-attachments/assets/be215eef-9855-42c1-acb4-78ae93d7b2e7" />


