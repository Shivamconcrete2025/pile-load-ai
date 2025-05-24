import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ----------------------
# Generate Synthetic Training Data
# ----------------------

@st.cache_data
def generate_and_train_model():
    np.random.seed(42)
    n_samples = 200
    data = {
        'pile_length': np.random.uniform(10, 30, n_samples),
        'pile_diameter': np.random.uniform(0.3, 1.2, n_samples),
        'spt_n_value': np.random.randint(10, 50, n_samples),
        'friction_angle': np.random.uniform(25, 40, n_samples),
        'pile_type': np.random.choice(['Driven', 'Bored'], n_samples)
    }

    df = pd.DataFrame(data)
    df['load_capacity'] = (
        100 * df['pile_length'] * df['pile_diameter'] +
        50 * df['spt_n_value'] +
        30 * df['friction_angle'] +
        np.where(df['pile_type'] == 'Driven', 500, -200) +
        np.random.normal(0, 1000, n_samples)
    )

    le = LabelEncoder()
    df['pile_type_encoded'] = le.fit_transform(df['pile_type'])

    X = df[['pile_length', 'pile_diameter', 'spt_n_value', 'friction_angle', 'pile_type_encoded']]
    y = df['load_capacity']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le

# Load model
model, label_encoder = generate_and_train_model()

# ----------------------
# Streamlit UI
# ----------------------

st.title("AI-Powered Pile Load Predictor 🏗️")
st.markdown("Predict pile ultimate load capacity and visualize the estimated load-settlement behavior.")

st.header("🔧 Enter Pile Parameters")

pile_length = st.slider("Pile Length (m)", 5.0, 40.0, 20.0)
pile_diameter = st.slider("Pile Diameter (m)", 0.3, 2.0, 0.6)
spt_n_value = st.slider("SPT-N Value", 5, 100, 30)
friction_angle = st.slider("Soil Friction Angle (°)", 20.0, 45.0, 35.0)
pile_type = st.selectbox("Pile Type", ['Driven', 'Bored'])

# Predict and Plot
if st.button("Predict and Show Curve"):
    # Encode and predict
    pile_type_encoded = label_encoder.transform([pile_type])[0]
    input_features = np.array([[pile_length, pile_diameter, spt_n_value, friction_angle, pile_type_encoded]])
    predicted_capacity = model.predict(input_features)[0]

    st.success(f"🧠 Predicted Ultimate Load Capacity: **{predicted_capacity:.2f} kN**")

    # Simulate a load-settlement curve
    loads = np.linspace(0, predicted_capacity * 1.2, 50)
    settlement = 0.05 * (loads / predicted_capacity) + 0.5 * (loads / predicted_capacity) ** 3  # synt**_*_*
