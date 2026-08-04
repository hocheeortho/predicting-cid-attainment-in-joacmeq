from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

# ------------------------------------------------
# Input utilities
# ------------------------------------------------
def optional_number_input(
    label,
    key,
    help_text=None,
    placeholder="Leave blank if unavailable",
):
    """
    Return a numeric value or None when the field is left blank.
    """
    value = st.text_input(
        label,
        value="",
        key=key,
        placeholder=placeholder,
        help=help_text,
    )

    if value.strip() == "":
        return None

    try:
        return float(value)
    except ValueError:
        st.warning(f"Please enter a numeric value for '{label}'.")
        return None


# ------------------------------------------------
# Input form
# ------------------------------------------------
st.header("Preoperative Patient Data")

input_values = {}


# ================================================================
# 1. Patient characteristics
# ================================================================
with st.expander("1. Patient characteristics", expanded=True):

    age = optional_number_input(
        "Age (years)",
        key="age",
    )

    sex_label = st.selectbox(
        "Sex",
        options=["Not entered"] + list(bundle["sex_options"].keys()),
    )

    diagnosis_label = st.selectbox(
        "Diagnosis",
        options=["Not entered"] + list(bundle["diagnosis_options"].keys()),
    )

    operation_label = st.selectbox(
        "Surgical procedure",
        options=["Not entered"] + list(bundle["operation_options"].keys()),
    )

    input_values["age"] = age

    input_values["sex"] = (
        None
        if sex_label == "Not entered"
        else bundle["sex_options"][sex_label]
    )

    input_values["diagnosis"] = (
        None
        if diagnosis_label == "Not entered"
        else bundle["diagnosis_options"][diagnosis_label]
    )

    input_values["operation"] = (
        None
        if operation_label == "Not entered"
        else bundle["operation_options"][operation_label]
    )


# ================================================================
# 2. JOACMEQ item responses
# ================================================================
JOACMEQ_ITEMS = [
    "Q1-1", "Q1-2", "Q1-3", "Q1-4",
    "Q2-1", "Q2-2", "Q2-3",
    "Q3-1", "Q3-2", "Q3-3", "Q3-4", "Q3-5",
    "Q4-1", "Q4-2", "Q4-3", "Q4-4",
    "Q5-1", "Q5-2", "Q5-3", "Q5-4",
    "Q5-5", "Q5-6", "Q5-7", "Q5-8",
]

with st.expander("2. JOACMEQ item responses"):

    st.caption(
        "Enter each response using the same numeric coding as the "
        "original JOACMEQ dataset. Fields may be left blank."
    )

    columns = st.columns(3)

    for index, item in enumerate(JOACMEQ_ITEMS):
        with columns[index % 3]:
            input_values[item] = optional_number_input(
                item,
                key=f"joacmeq_{item}",
            )


# ================================================================
# 3. VAS scores
# ================================================================
VAS_FEATURES = [
    "VAS pain or stiffness in neck",
    "VAS tightness in chest",
    "VAS pain or numbness in arms or hands",
    "VAS pain or numbness from chest to toe",
]

with st.expander("3. Visual analog scale scores"):

    st.caption(
        "Enter the preoperative VAS scores using the same scale as "
        "the development dataset."
    )

    for feature in VAS_FEATURES:
        input_values[feature] = optional_number_input(
            feature,
            key=f"vas_{feature}",
        )


# ================================================================
# 4. JOACMEQ domain scores
# ================================================================
DOMAIN_FEATURES = [
    "Cervical spine function",
    "Upper extremity function",
    "Lower extremity function",
    "Bladder function",
    "Quality of life",
]

with st.expander("4. JOACMEQ domain scores"):

    st.caption(
        "Enter preoperative domain scores from 0 to 100. "
        "Fields may be left blank."
    )

    columns = st.columns(2)

    for index, feature in enumerate(DOMAIN_FEATURES):
        with columns[index % 2]:
            input_values[feature] = optional_number_input(
                feature,
                key=f"domain_{feature}",
                help_text="Expected range: 0–100",
            )


# ================================================================
# 5. EQ-5D
# ================================================================
EQ5D_FEATURES = [
    "EQ-5D mobility",
    "EQ-5D self-care",
    "EQ-5D usual activities",
    "EQ-5D pain or discomfort",
    "EQ-5D anxiety or depression",
    "EQ-5D value",
]

with st.expander("5. EQ-5D"):

    st.caption(
        "Enter the five EQ-5D-5L item responses and the index value."
    )

    columns = st.columns(2)

    for index, feature in enumerate(EQ5D_FEATURES):
        with columns[index % 2]:
            input_values[feature] = optional_number_input(
                feature,
                key=f"eq5d_{feature}",
            )


# ================================================================
# 6. SF-8
# ================================================================
SF8_FEATURES = [
    "SF-8-1", "SF-8-2", "SF-8-3", "SF-8-4",
    "SF-8-5", "SF-8-6", "SF-8-7", "SF-8-8",
]

with st.expander("6. SF-8 responses"):

    st.caption(
        "Enter each SF-8 response using the same numeric coding as "
        "the development dataset."
    )

    columns = st.columns(2)

    for index, feature in enumerate(SF8_FEATURES):
        with columns[index % 2]:
            input_values[feature] = optional_number_input(
                feature,
                key=f"sf8_{feature}",
            )


# ------------------------------------------------
# Input status
# ------------------------------------------------
entered_count = sum(
    value is not None
    for value in input_values.values()
)

total_count = len(bundle["raw_feature_names"])

st.info(
    f"{entered_count} of {total_count} predictors have been entered. "
    "Missing predictor values are permitted."
)

# ------------------------------------------------
# Prediction utilities
# ------------------------------------------------
def normalize_category(value):
    """Convert categorical codes to the format used during training."""
    if value is None or pd.isna(value):
        return "missing"

    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def prepare_input_dataframe(input_dict):
    """
    Convert one patient's input into a one-row DataFrame using the
    same preprocessing applied during model training.
    """
    row = {}

    for feature in bundle["raw_feature_names"]:
        value = input_dict.get(feature)

        if feature in bundle["categorical_columns"]:
            row[feature] = normalize_category(value)
        else:
            row[feature] = np.nan if value is None else float(value)

    raw_df = pd.DataFrame([row])

    encoded_df = pd.get_dummies(
        raw_df,
        columns=bundle["categorical_columns"],
        drop_first=False,
        dtype=float,
    )

    return encoded_df


def predict_target_probability(target_data, encoded_input):
    """
    Generate the ensemble probability for one outcome.
    """
    required_columns = target_data["encoded_columns"]

    X = encoded_input.reindex(
        columns=required_columns,
        fill_value=0.0,
    )

    base_probabilities = []

    for model_name in target_data["base_model_order"]:
        model_info = target_data["base_models"][model_name]

        X_model = X.copy()

        # Median imputation for RF, SVM, ENLR, and LR
        median_values = model_info.get("median_values")

        if median_values is not None:
            X_model = X_model.fillna(median_values)

        # Standardization for SVM, ENLR, and LR
        scaler = model_info.get("scaler")

        if scaler is not None:
            X_model = scaler.transform(X_model)

        model = model_info["model"]

        raw_probability = model.predict_proba(X_model)[:, 1]

        # Apply the OOF-fitted isotonic calibrator
        calibrator = model_info.get("calibrator")

        if calibrator is not None:
            calibrated_probability = calibrator.predict(
                raw_probability
            )[0]
        else:
            calibrated_probability = raw_probability[0]

        base_probabilities.append(calibrated_probability)

    meta_features = np.array(
        base_probabilities,
        dtype=float,
    ).reshape(1, -1)

    ensemble_probability = target_data[
        "meta_model"
    ].predict_proba(meta_features)[0, 1]

    return float(ensemble_probability)


# ------------------------------------------------
# Prediction
# ------------------------------------------------
st.divider()

predict_button = st.button(
    "Predict CID attainment",
    type="primary",
    use_container_width=True,
)

if predict_button:

    if entered_count == 0:
        st.warning(
            "Please enter at least one predictor before running "
            "the prediction."
        )

    else:
        encoded_input = prepare_input_dataframe(input_values)

        prediction_results = []

        for target_name, target_data in bundle["targets"].items():

            probability = predict_target_probability(
                target_data,
                encoded_input,
            )

            display_name = target_data.get(
                "display_name",
                target_name.replace("MCID ", "", 1),
            )

            prediction_results.append({
                "Outcome": display_name,
                "Probability": probability,
                "Probability (%)": probability * 100,
            })

        results_df = pd.DataFrame(prediction_results)

        st.header("Predicted probability of CID attainment")

        # ------------------------------------------------
        # Percentage cards
        # ------------------------------------------------
        for start_index in range(0, len(results_df), 3):
            card_columns = st.columns(3)

            subset = results_df.iloc[
                start_index:start_index + 3
            ]

            for column, (_, result) in zip(
                card_columns,
                subset.iterrows(),
            ):
                with column:
                    st.metric(
                        label=result["Outcome"],
                        value=f"{result['Probability (%)']:.1f}%",
                    )

        # ------------------------------------------------
        # Bar chart
        # ------------------------------------------------
        chart_df = results_df[
            ["Outcome", "Probability (%)"]
        ].set_index("Outcome")

        st.subheader("Probability overview")
        st.bar_chart(chart_df)

        # ------------------------------------------------
        # Results table
        # ------------------------------------------------
        st.subheader("Prediction results")

        display_df = results_df[
            ["Outcome", "Probability (%)"]
        ].copy()

        display_df["Probability (%)"] = display_df[
            "Probability (%)"
        ].map(lambda value: f"{value:.1f}%")

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These estimates are generated by the stacked ensemble "
            "model. They should be interpreted as model-based "
            "probabilities rather than guaranteed individual outcomes."
        )
