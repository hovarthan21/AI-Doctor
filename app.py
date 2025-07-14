import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -------------------- Load Required Files --------------------
# Load trained model
with open("model/doctor_model.pkl", "rb") as f:
    doctor_model = pickle.load(f)

# Load feature names
with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load medical data
medical_df = pd.read_csv("cleaned_medical_data_with_food_updated.csv")
medical_df.columns = medical_df.columns.str.strip().str.lower()

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="AI Doctor", layout="wide")
st.title("🤖 AI Doctor - Smart Pain Diagnosis")

# -------------------- Navigation Tabs --------------------
page = st.sidebar.radio("📌 Navigate", ["🏠 Patient Info", "💡 Diagnosis"])

# -------------------- Page 1: Patient Info --------------------
if page == "🏠 Patient Info":
    st.header("👤 Patient Information")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    city = st.text_input("City")
    state = st.text_input("State")
    country = st.text_input("Country")

    if st.button("➡ Proceed to Diagnosis"):
        if name:
            # Save patient data
            patient_data = pd.DataFrame([[name, age, gender, city, state, country]],
                                        columns=["Name", "Age", "Gender", "City", "State", "Country"])
            file_path = "patient_records.xlsx"
            if os.path.exists(file_path):
                existing = pd.read_excel(file_path)
                patient_data = pd.concat([existing, patient_data], ignore_index=True)
            patient_data.to_excel(file_path, index=False)

            st.session_state["patient"] = name
            st.success("✅ Patient info saved. Go to Diagnosis tab.")
        else:
            st.warning("⚠ Please enter patient name.")

# -------------------- Page 2: Diagnosis --------------------
elif page == "💡 Diagnosis":
    st.header("🩺 Smart Diagnosis System")

    st.sidebar.header("📍 Pain Info")
    pain_area = st.sidebar.selectbox("Where is your pain?", [
        "Head", "Chest", "Abdomen", "Back", "Leg", "Arm", "Neck", "Shoulder", "Eye", "Ear", "Throat"])
    pain_level = st.sidebar.slider("Pain Level (0-5)", 0, 5, 2)
    pain_time = st.sidebar.selectbox("When does the pain occur?", ["Morning", "Afternoon", "Evening", "Night", "Always"])

    st.subheader("🧠 Select Your Symptoms")
    col1, col2 = st.columns(2)
    half = len(feature_names) // 2
    symptoms_1 = col1.multiselect("Symptoms - Part 1", feature_names[:half])
    symptoms_2 = col2.multiselect("Symptoms - Part 2", feature_names[half:])
    selected_symptoms = symptoms_1 + symptoms_2

    if st.button("🔍 Diagnose"):
        if not selected_symptoms:
            st.error("⚠ Please select at least one symptom.")
        else:
            # Create input vector
            input_dict = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in feature_names}
            input_df = pd.DataFrame([input_dict])

            # Predict disease
            predicted_disease = doctor_model.predict(input_df)[0]
            st.subheader(f"🧬 Predicted Disease: *{predicted_disease.title()}*")

            # Lookup additional information
            row = medical_df[medical_df["disease"].str.strip().str.lower() == predicted_disease.strip().lower()]
            if not row.empty:
                row = row.iloc[0]
                cause = row.get('causes', '')
                medicine = row.get('medicine', '')

                st.markdown(f"📄 **Cause:** {cause if cause else 'Not available'}")
                st.markdown(f"💊 **Recommended Medicine:** {medicine if medicine else 'Please consult a doctor or take general pain relievers'}")
            else:
                st.info("💊 **AI Suggestion:** Based on symptoms and disease, consult your Doctor or take medicine for your respective disease.")
                            
            with st.expander("📍 Pain Details"):
                st.markdown(f"- *Location:* {pain_area}")
                st.markdown(f"- *Level:* {pain_level}")
                st.markdown(f"- *Time:* {pain_time}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Built with ❤ by Hovarthan | AI Doctor © 2025")

# -------------------- Style --------------------
st.markdown("""
    <style>
        .block-container { padding: 1rem 2rem; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)
