import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── PAGE CONFIG
st.set_page_config(page_title="Insurance Charges Predictor", page_icon="🏥", layout="centered")

# ── LOAD ALL ARTIFACTS
@st.cache_resource
def load_artifacts():
    model         = pickle.load(open('modeldone (1).pkl', 'rb'))
    scaler        = pickle.load(open('scaler.pkl', 'rb'))
    feature_cols  = pickle.load(open('feature_cols.pkl', 'rb'))
    encoders = {
        'gender':                 pickle.load(open('gender_encoder.pkl', 'rb')),
        'smoker':                 pickle.load(open('smoker_encoder.pkl', 'rb')),
        'medical_history':        pickle.load(open('medical_history_encoder.pkl', 'rb')),
        'region':                 pickle.load(open('region_encoder.pkl', 'rb')),
        'family_medical_history': pickle.load(open('family_medical_history_encoder.pkl', 'rb')),
        'exercise_frequency':     pickle.load(open('exercise_frequency_encoder.pkl', 'rb')),
        'occupation':             pickle.load(open('occupation_encoder.pkl', 'rb')),
        'coverage_level':         pickle.load(open('coverage_level_encoder.pkl', 'rb')),
    }
    return model, scaler, feature_cols, encoders

model, scaler, feature_cols, encoders = load_artifacts()

# ── TITLE
st.title("🏥 Insurance Charges Predictor")
st.markdown("Fill in the details below to predict your insurance charges.")
st.divider()

# ── INPUT FORM
col1, col2 = st.columns(2)

with col1:
    age                    = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi                    = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children               = st.number_input("No. of Children", min_value=0, max_value=10, value=0)
    gender                 = st.selectbox("Gender", encoders['gender'].classes_)
    smoker                 = st.selectbox("Smoker", encoders['smoker'].classes_)

with col2:
    region                 = st.selectbox("Region", encoders['region'].classes_)
    medical_history        = st.selectbox("Medical History", encoders['medical_history'].classes_)
    family_medical_history = st.selectbox("Family Medical History", encoders['family_medical_history'].classes_)
    exercise_frequency     = st.selectbox("Exercise Frequency", encoders['exercise_frequency'].classes_)
    occupation             = st.selectbox("Occupation", encoders['occupation'].classes_)
    coverage_level         = st.selectbox("Coverage Level", encoders['coverage_level'].classes_)

st.divider()

# ── PREDICT
if st.button("🔮 Predict Charges", use_container_width=True):

    # 1. ENCODE using saved LabelEncoders
    gender_enc   = encoders['gender'].transform([gender])[0]
    smoker_enc   = encoders['smoker'].transform([smoker])[0]
    region_enc   = encoders['region'].transform([region])[0]
    med_enc      = encoders['medical_history'].transform([medical_history])[0]
    fam_med_enc  = encoders['family_medical_history'].transform([family_medical_history])[0]
    exer_enc     = encoders['exercise_frequency'].transform([exercise_frequency])[0]
    occ_enc      = encoders['occupation'].transform([occupation])[0]
    cov_enc      = encoders['coverage_level'].transform([coverage_level])[0]

    # 2. INTERACTION FEATURES — same as training
    smoker_x_coverage = smoker_enc * cov_enc
    smoker_x_med      = smoker_enc * med_enc
    age_x_smoker      = age        * smoker_enc
    bmi_x_smoker      = bmi        * smoker_enc
    coverage_x_med    = cov_enc    * med_enc
    fam_med_x_smoker  = fam_med_enc * smoker_enc

    # 3. BUILD DATAFRAME
    input_dict = {
        'age':                    age,
        'bmi':                    bmi,
        'children':               children,
        'gender':                 gender_enc,
        'smoker':                 smoker_enc,
        'region':                 region_enc,
        'medical_history':        med_enc,
        'family_medical_history': fam_med_enc,
        'exercise_frequency':     exer_enc,
        'occupation':             occ_enc,
        'coverage_level':         cov_enc,
        'smoker_x_coverage':      smoker_x_coverage,
        'smoker_x_med':           smoker_x_med,
        'age_x_smoker':           age_x_smoker,
        'bmi_x_smoker':           bmi_x_smoker,
        'coverage_x_med':         coverage_x_med,
        'fam_med_x_smoker':       fam_med_x_smoker,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_cols]  # match training column order

    # 4. SCALE — same scaler used on X_train
    input_scaled = scaler.transform(input_df)

    # 5. PREDICT
    prediction = model.predict(input_scaled)[0]

    # 6. DISPLAY
    st.success(f"### 💰 Estimated Insurance Charges: **${prediction:,.2f}**")

    st.markdown("#### 📊 Your Risk Summary")
    r1, r2, r3 = st.columns(3)
    r1.metric("Smoker",          smoker.upper())
    r2.metric("Coverage Level",  coverage_level)
    r3.metric("Medical History", medical_history)