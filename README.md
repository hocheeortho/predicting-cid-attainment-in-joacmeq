# Predicting CID Attainment in JOACMEQ

This repository contains a web application for estimating the probability of achieving a clinically important difference (CID) after cervical spine surgery.

The application provides individualized probabilities for five JOACMEQ domains and four visual analog scale outcomes.

## Model

The prediction system uses six base models:

- LightGBM
- XGBoost
- Random Forest
- Support Vector Machine
- Elastic-net logistic regression
- Logistic regression

The outputs of the six models are combined using an L2-regularized logistic regression stacking model.

The final models were trained using the development cohort from DAIRO and JCHO.

## Application

The web application accepts preoperative patient characteristics and questionnaire data, including:

- Age
- Sex
- Diagnosis
- Surgical procedure
- JOACMEQ item responses
- JOACMEQ domain scores
- Visual analog scale scores
- EQ-5D responses
- SF-8 responses

Missing predictor values are allowed.

## Outcomes

The application estimates CID attainment probabilities for:

1. Cervical spine function
2. Upper extremity function
3. Lower extremity function
4. Bladder function
5. Quality of life
6. Pain or stiffness in the neck
7. Tightness in the chest
8. Pain or numbness in the arms or hands
9. Pain or numbness from the chest to the toes

## Disclaimer

This application is intended for research and informational purposes only. It is not intended to replace clinical judgment or provide a definitive prediction for an individual patient.
