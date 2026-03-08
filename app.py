import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load model & data ─────────────────────────────────────
model        = joblib.load('model/diagnosbot_model.pkl')
le           = joblib.load('model/label_encoder.pkl')
all_symptoms = joblib.load('model/symptoms_list.pkl')
desc_df      = pd.read_csv('data/symptom_description.csv')
prec_df      = pd.read_csv('data/symptom_precaution.csv')
sev_df       = pd.read_csv('data/symptom_severity.csv')

def get_description(disease):
    row = desc_df[desc_df['Disease'] == disease]
    return row['Description'].values[0] if not row.empty else "No description available."

def get_precautions(disease):
    row = prec_df[prec_df['Disease'] == disease]
    if row.empty: return []
    return [row[f'Precaution_{i}'].values[0] for i in range(1,5) if pd.notna(row[f'Precaution_{i}'].values[0])]

def get_severity(symptoms):
    total = 0
    for s in symptoms:
        row = sev_df[sev_df['Symptom'] == s]
        if not row.empty:
            total += row['weight'].values[0]
    return total

def predict(symptoms):
    vec = np.zeros(len(all_symptoms))
    for s in symptoms:
        if s in all_symptoms:
            vec[all_symptoms.index(s)] = 1
    pred = model.predict([vec])[0]
    proba = model.predict_proba([vec])[0]
    confidence = round(max(proba) * 100, 1)
    disease = le.inverse_transform([pred])[0]
    return disease, confidence

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="DiagnosBot",
    page_icon="🩺",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#2C7BE5;'>🩺 DiagnosBot</h1>
    <p style='text-align:center; color:gray;'>AI-Powered Disease Diagnosis Assistant</p>
    <hr>
""", unsafe_allow_html=True)

st.warning("⚠️ This tool is for educational purposes only. Always consult a real doctor.")

# ── Symptom selection ─────────────────────────────────────
st.subheader("📋 Select Your Symptoms")
selected = st.multiselect(
    "Choose all symptoms you are experiencing:",
    options=all_symptoms,
    placeholder="Type or select symptoms..."
)

# ── Diagnose button ───────────────────────────────────────
if st.button("🔍 Diagnose", use_container_width=True):
    if len(selected) < 2:
        st.error("❌ Please select at least 2 symptoms.")
    else:
        disease, confidence = predict(selected)
        severity = get_severity(selected)

        # Severity level
        if severity <= 10:
            sev_label, sev_color = "Mild 🟢", "green"
        elif severity <= 20:
            sev_label, sev_color = "Moderate 🟡", "orange"
        else:
            sev_label, sev_color = "Severe 🔴", "red"

        st.markdown("---")
        st.markdown(f"### 🏥 Diagnosed Disease: **{disease}**")

        col1, col2 = st.columns(2)
        col1.metric("🎯 Confidence", f"{confidence}%")
        col2.metric("⚠️ Severity", sev_label)

        # Description
        st.markdown("#### 📖 About this Disease")
        st.info(get_description(disease))

        # Precautions
        precautions = get_precautions(disease)
        if precautions:
            st.markdown("#### 💊 Recommended Precautions")
            for i, p in enumerate(precautions, 1):
                st.success(f"{i}. {p.capitalize()}")

        st.markdown("---")
        st.markdown("<p style='text-align:center;color:gray;'>Always consult a certified medical professional.</p>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("<br><hr><p style='text-align:center;color:gray;'>Built by M Saqib Bilal | DiagnosBot v1.0</p>", unsafe_allow_html=True)
