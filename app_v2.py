import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ── Load Model ─────────────────────────────────────────────
model    = joblib.load('model_v2/diagnosbot_model.pkl')
le       = joblib.load('model_v2/label_encoder.pkl')
symptoms = joblib.load('model_v2/symptoms_list.pkl')

# ── Load support data ──────────────────────────────────────
desc_df  = pd.read_csv('data/symptom_description.csv')
prec_df  = pd.read_csv('data/symptom_precaution.csv')
sev_df   = pd.read_csv('data/symptom_severity.csv')

def get_description(disease):
    row = desc_df[desc_df['Disease'].str.lower() == disease.lower()]
    return row['Description'].values[0] if not row.empty else "Consult a doctor for detailed information."

def get_precautions(disease):
    row = prec_df[prec_df['Disease'].str.lower() == disease.lower()]
    if row.empty: return []
    return [row[f'Precaution_{i}'].values[0] for i in range(1,5)
            if pd.notna(row[f'Precaution_{i}'].values[0])]

def get_severity(selected_symptoms):
    total = 0
    for s in selected_symptoms:
        row = sev_df[sev_df['Symptom'] == s]
        if not row.empty:
            total += row['weight'].values[0]
    return total

def predict_top3(selected_symptoms):
    vec = np.zeros(len(symptoms))
    for s in selected_symptoms:
        if s in symptoms:
            vec[symptoms.index(s)] = 1
    proba   = model.predict_proba([vec])[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    results = []
    for idx in top3_idx:
        disease    = le.inverse_transform([idx])[0]
        confidence = round(proba[idx] * 100, 1)
        results.append((disease, confidence))
    return results

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="DiagnosBot v2.0",
    page_icon="🩺",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#2C7BE5;'>🩺 DiagnosBot <span style='color:#00c9b1;'>v2.0</span></h1>
    <p style='text-align:center; color:gray; font-size:16px;'>AI-Powered Disease Diagnosis | 41 Diseases | 132 Symptoms | 100% Accuracy</p>
    <hr>
""", unsafe_allow_html=True)

st.warning("⚠️ For educational purposes only. Always consult a certified medical professional.")

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Diagnose", "📊 Symptom Insights"])

# ── TAB 1: DIAGNOSIS ───────────────────────────────────────
with tab1:
    st.subheader("📋 Select Your Symptoms")
    st.caption(f"Choose from {len(symptoms)} available symptoms")

    # Search symptoms
    search = st.text_input("🔍 Search symptoms", placeholder="Type to filter symptoms...")
    filtered_symptoms = [s for s in symptoms if search.lower() in s.lower()] if search else symptoms
    clean_symptoms = [s.replace('_', ' ').title() for s in filtered_symptoms]
    symptom_map = dict(zip(clean_symptoms, filtered_symptoms))

    selected_display = st.multiselect(
        "Select all symptoms you are experiencing:",
        options=clean_symptoms,
        placeholder="Choose symptoms..."
    )
    selected = [symptom_map[s] for s in selected_display]

    col1, col2 = st.columns([1, 3])
    with col1:
        diagnose_btn = st.button("🔍 Diagnose Now", use_container_width=True, type="primary")
    with col2:
        if selected:
            st.info(f"✅ {len(selected)} symptom(s) selected")

    if diagnose_btn:
        if len(selected) < 2:
            st.error("❌ Please select at least 2 symptoms for accurate diagnosis.")
        else:
            top3 = predict_top3(selected)
            severity = get_severity(selected)
            disease, confidence = top3[0]

            # Severity level
            if severity <= 10:
                sev_label, sev_color = "Mild 🟢", "#00c9b1"
            elif severity <= 20:
                sev_label, sev_color = "Moderate 🟡", "#f0b429"
            else:
                sev_label, sev_color = "Severe 🔴", "#E53E3E"

            st.markdown("---")

            # ── Primary Result ─────────────────────────────
            st.markdown(f"## 🏥 Primary Diagnosis: **{disease}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Confidence", f"{confidence}%")
            col2.metric("⚠️ Severity", sev_label)
            col3.metric("🔬 Symptoms Analyzed", len(selected))

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Diagnosis Confidence %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "#2C7BE5"},
                    'steps': [
                        {'range': [0,  50], 'color': '#FED7D7'},
                        {'range': [50, 80], 'color': '#FEFCBF'},
                        {'range': [80, 100],'color': '#C6F6D5'},
                    ]
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Top 3 Predictions ──────────────────────────
            st.markdown("#### 🎯 Top 3 Possible Diagnoses")
            for i, (d, c) in enumerate(top3):
                emoji = ["🥇", "🥈", "🥉"][i]
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(int(c), text=f"{emoji} {d}")
                with col_b:
                    st.markdown(f"**{c}%**")

            # ── Disease Info ───────────────────────────────
            st.markdown("---")
            col_desc, col_prec = st.columns(2)

            with col_desc:
                st.markdown("#### 📖 About This Disease")
                st.info(get_description(disease))

            with col_prec:
                st.markdown("#### 💊 Recommended Precautions")
                precautions = get_precautions(disease)
                if precautions:
                    for i, p in enumerate(precautions, 1):
                        st.success(f"{i}. {p.capitalize()}")
                else:
                    st.info("Consult a doctor for specific precautions.")

            # ── Selected Symptoms Summary ──────────────────
            st.markdown("---")
            st.markdown("#### 🔬 Your Reported Symptoms")
            cols = st.columns(4)
            for i, s in enumerate(selected):
                cols[i % 4].markdown(f"• {s.replace('_', ' ').title()}")

# ── TAB 2: INSIGHTS ────────────────────────────────────────
with tab2:
    st.subheader("📊 Symptom & Disease Insights")

    # Feature importance chart
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top15 = sorted(zip(symptoms, importances), key=lambda x: x[1], reverse=True)[:15]
        sym_names = [s[0].replace('_', ' ').title() for s in top15]
        sym_vals  = [round(s[1]*100, 2) for s in top15]

        fig_imp = px.bar(
            x=sym_vals, y=sym_names,
            orientation='h',
            title="🔑 Top 15 Most Diagnostically Important Symptoms",
            labels={'x': 'Importance Score (%)', 'y': 'Symptom'},
            color=sym_vals,
            color_continuous_scale='Blues'
        )
        fig_imp.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Severity distribution
    st.markdown("#### ⚠️ Symptom Severity Distribution")
    fig_sev = px.histogram(
        sev_df, x='weight', nbins=10,
        title="Distribution of Symptom Severity Weights",
        color_discrete_sequence=['#2C7BE5']
    )
    st.plotly_chart(fig_sev, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
    <hr>
    <p style='text-align:center; color:gray;'>
    🩺 DiagnosBot v2.0 | 100% Accuracy | 41 Diseases | 132 Symptoms<br>
    Built by <strong>M Saqib Bilal</strong> | Open Source Medical AI
    </p>
""", unsafe_allow_html=True)
