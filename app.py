from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Predicting CID Attainment in JOACMEQ"
MODEL_PATH = Path("models/ensemble_bundle.joblib")

FEATURE_GROUPS = {
    "Patient characteristics": ["age", "sex", "operation"],
    "JOACMEQ item responses": [
        "Q1-1", "Q1-2", "Q1-3", "Q1-4",
        "Q2-1", "Q2-2", "Q2-3",
        "Q3-1", "Q3-2", "Q3-3", "Q3-4", "Q3-5",
        "Q4-1", "Q4-2", "Q4-3", "Q4-4",
        "Q5-1", "Q5-2", "Q5-3", "Q5-4", "Q5-5", "Q5-6", "Q5-7", "Q5-8",
    ],
    "VAS scores": [
        "VAS pain or stiffness in neck",
        "VAS tightness in chest",
        "VAS pain or numbness in arms or hands",
        "VAS pain or numbness from chest to toe",
    ],
    "JOACMEQ domain scores": [
        "Cervical spine function",
        "Upper extremity function",
        "Lower extremity function",
        "Bladder function",
        "Quality of life",
    ],
    "EQ-5D": [
        "EQ-5D mobility",
        "EQ-5D self-care",
        "EQ-5D usual activities",
        "EQ-5D pain or discomfort",
        "EQ-5D anxiety or depression",
        "EQ-5D value",
    ],
    "SF-8": [f"SF-8-{i}" for i in range(1, 9)],
}

TARGET_LABELS = [
    "Cervical spine function",
    "Upper extremity function",
    "Lower extremity function",
    "Bladder function",
    "Quality of life",
    "VAS pain or stiffness in neck",
    "VAS tightness in chest",
    "VAS pain or numbness in arms or hands",
    "VAS pain or numbness from chest to toe",
]


def parse_optional_number(value: str):
    text = value.strip()
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        st.error(f"Invalid numeric value: {value}")
        st.stop()


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_probabilities(bundle, raw_input: pd.DataFrame) -> pd.Series:
    required_keys = {"feature_columns", "models", "meta_models", "medians", "target_names"}
    missing = required_keys - set(bundle)
    if missing:
        raise KeyError(f"Model bundle is missing required keys: {sorted(missing)}")

    feature_columns = bundle["feature_columns"]
    medians = pd.Series(bundle["medians"], dtype=float)
    base_models = bundle["models"]
    meta_models = bundle["meta_models"]
    target_names = bundle["target_names"]

    x = raw_input.copy()
    x = pd.get_dummies(x, columns=["sex", "operation"], drop_first=False)
    x = x.reindex(columns=feature_columns, fill_value=0)

    output = {}
    for target in target_names:
        target_models = base_models[target]
        base_probabilities = []

        for model_name, artifact in target_models.items():
            model = artifact["model"]
            scaler = artifact.get("scaler")

            if model_name in {"LightGBM", "XGBoost"}:
                x_model = x.copy()
            else:
                x_model = x.copy().fillna(medians)

            if scaler is not None:
                x_model = scaler.transform(x_model)

            probability = float(model.predict_proba(x_model)[:, 1][0])
            base_probabilities.append(probability)

        meta_input = np.asarray(base_probabilities, dtype=float).reshape(1, -1)
        output[target] = float(meta_models[target].predict_proba(meta_input)[:, 1][0])

    return pd.Series(output, dtype=float)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Research-use application for estimating CID-attainment probabilities after cervical spine surgery. "
    "Blank fields are treated as missing values."
)

bundle = load_bundle()
if bundle is None:
    st.warning(
        "The trained ensemble bundle is not yet available. Add `models/ensemble_bundle.joblib` "
        "to enable prediction. The interface below can still be reviewed."
    )

with st.form("prediction_form"):
    values = {}

    for group_name, features in FEATURE_GROUPS.items():
        st.subheader(group_name)
        columns = st.columns(3)

        for index, feature in enumerate(features):
            column = columns[index % 3]
            with column:
                if feature == "sex":
                    selected = st.selectbox(
                        "Sex",
                        options=["Missing", "Male", "Female"],
                        key=feature,
                    )
                    values[feature] = np.nan if selected == "Missing" else selected
                elif feature == "operation":
                    selected = st.selectbox(
                        "Operation",
                        options=[
                            "Missing",
                            "Laminoplasty",
                            "Anterior spinal fusion",
                            "Posterior spinal fusion",
                            "Other",
                        ],
                        key=feature,
                    )
                    values[feature] = np.nan if selected == "Missing" else selected
                else:
                    raw = st.text_input(feature, value="", key=feature)
                    values[feature] = raw

    submitted = st.form_submit_button("Predict CID attainment")

if submitted:
    for key, value in list(values.items()):
        if key not in {"sex", "operation"}:
            values[key] = parse_optional_number(value)

    input_df = pd.DataFrame([values])

    if bundle is None:
        st.error("Prediction cannot be performed until the trained model bundle is added.")
    else:
        try:
            probabilities = predict_probabilities(bundle, input_df)
            result = pd.DataFrame(
                {
                    "Outcome": [name.replace("MCID ", "") for name in probabilities.index],
                    "Predicted CID-attainment probability": [f"{100 * p:.1f}%" for p in probabilities.values],
                }
            )
            st.subheader("Prediction results")
            st.dataframe(result, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.exception(exc)

st.divider()
st.markdown(
    "**Research-use notice:** This software is not a medical device and must not be used as the sole basis "
    "for diagnosis, prognosis, treatment selection, or patient counseling."
)
