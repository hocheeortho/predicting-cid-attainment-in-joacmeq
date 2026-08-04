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
def optional_choice_input(label, options, key):
    """
    Return an integer response or None when not entered.
    """
    selected = st.selectbox(
        label,
        options=["Not entered"] + options,
        key=key,
    )

    if selected == "Not entered":
        return None

    return int(selected)

# ------------------------------------------------
# Input form
# ------------------------------------------------
st.header("Preoperative Patient Data")

input_values = {}


# ================================================================
# 1. Patient characteristics
# ================================================================
with st.expander(
    "1. Patient characteristics",
    expanded=True,
):

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
JOACMEQ_RESPONSE_OPTIONS = {
    "Q1-1": [1, 2, 3],
    "Q1-2": [1, 2, 3],
    "Q1-3": [1, 2, 3],
    "Q1-4": [1, 2, 3],

    "Q2-1": [1, 2, 3],
    "Q2-2": [1, 2, 3],
    "Q2-3": [1, 2, 3, 4],

    "Q3-1": [1, 2, 3, 4, 5],
    "Q3-2": [1, 2, 3],
    "Q3-3": [1, 2, 3],
    "Q3-4": [1, 2, 3],
    "Q3-5": [1, 2, 3],

    "Q4-1": [1, 2, 3, 4, 5],
    "Q4-2": [1, 2, 3],
    "Q4-3": [1, 2, 3],
    "Q4-4": [1, 2, 3],

    "Q5-1": [1, 2, 3, 4, 5],
    "Q5-2": [1, 2, 3, 4, 5],
    "Q5-3": [1, 2, 3, 4, 5],
    "Q5-4": [1, 2, 3, 4, 5],
    "Q5-5": [1, 2, 3, 4, 5],
    "Q5-6": [1, 2, 3, 4, 5],
    "Q5-7": [1, 2, 3, 4, 5],
    "Q5-8": [1, 2, 3, 4, 5],
}


with st.expander("2. JOACMEQ item responses"):

    st.caption(
        "Select the response number for each JOACMEQ item. "
        "Domain scores are calculated automatically."
    )

    columns = st.columns(3)

    for index, (item, options) in enumerate(
        JOACMEQ_RESPONSE_OPTIONS.items()
    ):
        with columns[index % 3]:
            input_values[item] = optional_choice_input(
                label=item,
                options=options,
                key=f"joacmeq_{item}",
            )
def calculate_joacmeq_domain_scores(values):
    """
    Calculate the five JOACMEQ domain scores from item responses.

    A domain score is returned as None when one or more required
    responses for that domain are missing.
    """

    domain_definitions = {
        "Cervical spine function": {
            "weights": {
                "Q1-1": 20,
                "Q1-2": 10,
                "Q1-3": 15,
                "Q1-4": 5,
            },
            "minimum": 50,
            "range": 100,
        },

        "Upper extremity function": {
            "weights": {
                "Q1-4": 5,
                "Q2-1": 10,
                "Q2-2": 15,
                "Q2-3": 5,
                "Q3-1": 5,
            },
            "minimum": 40,
            "range": 95,
        },

        "Lower extremity function": {
            "weights": {
                "Q3-1": 10,
                "Q3-2": 10,
                "Q3-3": 15,
                "Q3-4": 5,
                "Q3-5": 5,
            },
            "minimum": 45,
            "range": 110,
        },

        "Bladder function": {
            "weights": {
                "Q4-1": 10,
                "Q4-2": 5,
                "Q4-3": 10,
                "Q4-4": 5,
            },
            "minimum": 30,
            "range": 80,
        },

        "Quality of life": {
            "weights": {
                "Q5-1": 3,
                "Q5-2": 2,
                "Q5-3": 2,
                "Q5-4": 5,
                "Q5-5": 4,
                "Q5-6": 3,
                "Q5-7": 2,
                "Q5-8": 3,
            },
            "minimum": 24,
            "range": 96,
        },
    }

    calculated_scores = {}

    for domain_name, definition in domain_definitions.items():
        required_items = definition["weights"]

        # Do not calculate the domain when a required item is missing
        if any(values.get(item) is None for item in required_items):
            calculated_scores[domain_name] = None
            continue

        weighted_sum = sum(
            float(values[item]) * weight
            for item, weight in required_items.items()
        )

        score = (
            weighted_sum - definition["minimum"]
        ) * 100 / definition["range"]

        # Protect against values outside the theoretical range
        score = float(np.clip(score, 0.0, 100.0))

        calculated_scores[domain_name] = score

    return calculated_scores


domain_scores = calculate_joacmeq_domain_scores(input_values)

# Add the automatically calculated scores to the model input
input_values.update(domain_scores)

with st.expander(
    "Calculated JOACMEQ domain scores",
    expanded=True,
):
    score_columns = st.columns(3)

    for index, (domain_name, score) in enumerate(
        domain_scores.items()
    ):
        with score_columns[index % 3]:
            if score is None:
                st.metric(
                    label=domain_name,
                    value="Not calculated",
                )
                st.caption("One or more required items are missing.")
            else:
                st.metric(
                    label=domain_name,
                    value=f"{score:.1f}",
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

with st.expander("3. Visual analog scale scores",
    expanded=True,):

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
# 4. EQ-5D
# ================================================================
EQ5D_ITEM_OPTIONS = {
    "EQ-5D mobility": [1, 2, 3],
    "EQ-5D self-care": [1, 2, 3],
    "EQ-5D usual activities": [1, 2, 3],
    "EQ-5D pain or discomfort": [1, 2, 3],
    "EQ-5D anxiety or depression": [1, 2, 3],
}

with st.expander("4. EQ-5D"):

    st.caption(
        "Select the response for each EQ-5D item. "
        "The EQ-5D index value is optional."
    )

    cols = st.columns(2)

    for i, (feature, options) in enumerate(EQ5D_ITEM_OPTIONS.items()):
        with cols[i % 2]:
            input_values[feature] = optional_choice_input(
                feature,
                options,
                key=f"eq5d_{feature}",
            )

    input_values["EQ-5D value"] = optional_number_input(
        "EQ-5D index value (optional)",
        key="eq5d_value",
        help_text=(
            "Enter the calculated EQ-5D index value if available. "
            "If left blank, the model will handle it as a missing value."
        ),
    )
    st.info(
        "The EQ-5D index value is optional. "
        "If left blank, the prediction model will automatically treat it as a missing value."
    )
# ================================================================
# 5. SF-8 responses
# ================================================================
SF8_RESPONSE_OPTIONS = {
    "SF-8-1": [1, 2, 3, 4, 5, 6],
    "SF-8-2": [1, 2, 3, 4, 5],
    "SF-8-3": [1, 2, 3, 4, 5],
    "SF-8-4": [1, 2, 3, 4, 5, 6],
    "SF-8-5": [1, 2, 3, 4, 5],
    "SF-8-6": [1, 2, 3, 4, 5],
    "SF-8-7": [1, 2, 3, 4, 5],
    "SF-8-8": [1, 2, 3, 4, 5],
}

with st.expander("5. SF-8 responses"):

    st.caption(
        "Select the response number for each SF-8 item."
    )

    columns = st.columns(2)

    for index, (feature, options) in enumerate(
        SF8_RESPONSE_OPTIONS.items()
    ):
        with columns[index % 2]:
            input_values[feature] = optional_choice_input(
                label=feature,
                options=options,
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
    row["EQ-5D value"] = np.nan
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
