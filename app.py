from pathlib import Path

import joblib
import streamlit as st


# ------------------------------------------------
# Page settings
# ------------------------------------------------
st.set_page_config(
    page_title="JOACMEQ CID Prediction",
    page_icon="🩺",
    layout="wide",
)


# ------------------------------------------------
# Load the trained model bundle
# ------------------------------------------------
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "ensemble_bundle.joblib"
)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()


# ------------------------------------------------
# Application title
# ------------------------------------------------
st.title("Prediction of CID Attainment After Cervical Spine Surgery")

st.write(
    "This application estimates the probability of achieving a "
    "clinically important difference in each JOACMEQ and VAS outcome."
)


# ------------------------------------------------
# Model information
# ------------------------------------------------
with st.expander("Model information"):
    st.write("Development sites:", ", ".join(bundle["development_sites"]))
    st.write("Number of outcomes:", len(bundle["targets"]))
    st.write("Base models:", ", ".join(bundle["base_model_order"]))


st.success("The prediction model was loaded successfully.")

st.header("Patient Information")

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=70
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

diagnosis = st.selectbox(
    "Diagnosis",
    [
        "CSM",
        "CSR / Disc herniation",
        "OPLL / OLF",
        "Other"
    ]
)

operation = st.selectbox(
    "Surgical procedure",
    [
        "Laminoplasty",
        "Posterior spinal fusion (PSF)",
        "Anterior spinal fusion (ASF)",
        "Combined ASF and PSF",
        "Other"
    ]
)
