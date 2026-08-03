# Predicting CID Attainment in JOACMEQ

A research-use web application for estimating the probability of clinically important difference (CID) attainment after cervical spine surgery from preoperative JOACMEQ-related variables.

## Scope

- **Training cohort:** DAIRO + JCHO
- **External validation cohort:** HANDAI
- **Interface language:** English
- **Outputs:** Predicted CID-attainment probabilities for five JOACMEQ domains and four VAS outcomes
- **Primary model:** Stacking ensemble using LightGBM, XGBoost, Random Forest, support vector machine, elastic-net logistic regression, and logistic regression

## Missing values

Missing inputs are allowed. LightGBM and XGBoost use their native missing-value handling. Random Forest, support vector machine, elastic-net logistic regression, and logistic regression use medians estimated from the development cohort.

## Repository status

The application scaffold is included, but trained model artifacts are not yet committed. The app will become operational after the exported model bundle is added to `models/ensemble_bundle.joblib`.

## Local use

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Research-use notice

This software is provided for research and academic demonstration only. It is not a medical device and must not be used as the sole basis for clinical decision-making. No patient-level source data are included in this public repository.
