import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------------------
# Train AI Model (Simulated Data)
# ---------------------
@st.cache_data
def train_model():
    np.random.seed(42)
    n_samples = 500
    data = {
        'max_velocity': np.random.normal(4, 1, n_samples),
        'signal_duration': np.random.normal(20, 5, n_samples),
        'reflected_peaks': np.random.randint(0, 5, n_samples),
        'energy': np.random.normal(80, 20, n_samples),
        'defect_depth': np.random.uniform(0, 10, n_samples)
    }

    labels = []
    for i in range(n_samples):
        if data['reflected_peaks'][i] >= 3 or data['defect_depth'][i] > 6:
            labels.append(2)  # Defective
        elif data['reflected_peaks'][i] == 2:
            labels.append(1)  # Possible defect
        else:
            labels.append(0)  # Good

    df = pd.DataFrame(data)
    df['condition'] = labels

    X = df.drop(columns='condition')
    y = df['condition']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# Load model
model = train_model()

# ---------------------
# Streamlit UI
# ---------------------

st.set_page_config(page_title="Pile Integrity Test Analyzer", layout="centered")
st.title("🧠 Pile Integrity Test Analyzer")
st.markdown("Enter PIT summary parameters to predict pile condition.")

# Input form
with st.form("PIT Input Form"):
    col1, col2 = st.columns(2)
    with col1:
        max_velocity = st.number_input("Max Velocity (m/s)", 0.0, 20.0, 4.0)
        signal_duration = st.number_input("Signal Duration (ms)", 1.0, 100.0, 20.0)
        reflected_peaks = st.slider("Reflected Peaks", 0, 5, 1)
    with col2:
        energy = st.number_input("Energy (kN·ms)", 0.0, 200.0, 80.0)
        defect_depth = st.slider("Defect Depth (m)", 0.0, 15.0, 2.0)

    submitted = st.form_submit_button("Predict Condition")

# Predict
if submitted:
    input_data = np.array([[max_velocity, signal_duration, reflected_peaks, energy, defect_depth]])
    prediction = model.predict(input_data)[0]
    label_map = {0: "✅ Good", 1: "⚠️ Possible Defect", 2: "❌ Defective"}
    st.subheader("Prediction Result")
    st.success(f"Predicted Condition: **{label_map[prediction]}**")

    # Show explanation
    st.markdown("### 🧾 Input Summary")
    st.json({
        "Max Velocity": max_velocity,
        "Signal Duration": signal_duration,
        "Reflected Peaks": reflected_peaks,
        "Energy": energy,
        "Defect Depth": defect_depth
    })

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and AI")
