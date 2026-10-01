import os

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Kano Crop Yield Prediction System",
    page_icon="🌱",
    layout="centered"
)

# --------------------------------------------------
# Model path
# --------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "kano_crop_yield_neural_network.pkl"
)

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🌱 Kano Crop Yield Prediction System")

st.markdown(
    """
    **Machine learning-based crop yield prediction for selected crops in Kano State.**

    Enter the available farm information below to estimate crop yield.
    """
)

st.divider()

# --------------------------------------------------
# Prediction form
# --------------------------------------------------
st.subheader("Farm Information")

with st.form("prediction_form"):

    crop_name = st.selectbox(
        "Crop",
        [
            "BEANS/COWPEA",
            "GUINEA CORN/SORGHUM",
            "MAIZE",
            "MILLET/MAIWA",
            "RICE"
        ],
        help="Select the crop for which you want to predict yield."
    )

    planted_area = st.number_input(
        "Planted Area (m²)",
        min_value=0.0,
        step=1.0,
        help="Enter the cultivated area in square metres (m²)."
    )

    irrigation_method = st.selectbox(
        "Irrigation Method",
        [
            "Bubbler irrigation",
            "Drip irrigation",
            "Equipped flood recession cultivation",
            "Equipped wetland and inland valley bottoms",
            "Not reported",
            "Other ( Specify)",
            "Spate irrigation",
            "Spray or microsprinkler irrigation",
            "Sprinkler irrigation",
            "Surface irrigation (flooding, furrows)"
        ]
    )

    water_source = st.selectbox(
        "Water Source",
        [
            "Municipal water supply or other water network",
            "Not reported",
            "Off-farm surface water (lakes, rivers, watercourses)",
            "On-farm ground water",
            "On-farm surface water",
            "Other ( Specify)",
            "Reservoir (used to avoid flooding)",
            "Treated waste water"
        ]
    )

    cropping_pattern = st.selectbox(
        "Cropping Pattern",
        [
            "MIXED / INTERCROPPED",
            "PURE STAND"
        ]
    )

    submitted = st.form_submit_button(
        "🌾 Predict Crop Yield",
        type="primary"
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

    if planted_area <= 0:
        st.error("Please enter a planted area greater than zero.")

    else:

        input_data = pd.DataFrame([{
            "crop_name": crop_name,
            "planted_area": planted_area,
            "irrigation_method": irrigation_method,
            "water_source": water_source,
            "cropping_pattern": cropping_pattern
        }])

        prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("Prediction Result")

        st.success("Prediction completed successfully.")

        yield_kg_ha = prediction * 10000

        st.metric(
            label="Predicted Crop Yield (kg/m²)",
            value=f"{prediction:.4f}"
        )

        st.metric(
            label="Equivalent Crop Yield (kg/ha)",
            value=f"{yield_kg_ha:,.2f}"
        )

        st.caption(
            "The primary prediction uses the same yield representation as "
            "the NASS dataset. The kg/ha value is a converted equivalent."
        )

# --------------------------------------------------
# About the model
# --------------------------------------------------
st.divider()

with st.expander("About the Model"):

    st.write(
        "The system uses a Neural Network regression model "
        "(MLPRegressor) trained using selected NASS crop records from Kano State."
    )

    st.write("**Model architecture:** 64 and 32 hidden neurons")

    st.write("**Input features:**")
    st.write(
        "- Crop name\n"
        "- Planted area\n"
        "- Irrigation method\n"
        "- Water source\n"
        "- Cropping pattern"
    )

    st.write("**Test performance:**")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "0.094606")
    col2.metric("RMSE", "0.153376")
    col3.metric("R²", "0.098517")

# --------------------------------------------------
# How the system works
# --------------------------------------------------
with st.expander("How the System Works"):

    st.write(
        "1. Enter the farm information.\n\n"
        "2. The system processes the selected inputs using the same "
        "preprocessing pipeline used during model training.\n\n"
        "3. The trained Neural Network generates a predicted crop yield.\n\n"
        "4. The prediction is displayed immediately on the screen."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Final-Year Computer Science Project | "
    "Northwest University Kano"
)
