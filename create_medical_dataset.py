from groq import Groq
import pandas as pd
import json
import time

client = Groq(api_key="YOUR_KEY_HERE"

# Medically accurate disease-symptom mappings
# Based on WHO and medical literature
MEDICAL_DATA = {
    "Malaria": {
        "primary": ["high_fever", "chills", "sweating", "headache", "nausea", "vomiting", "muscle_pain", "fatigue"],
        "secondary": ["loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever"],
        "rare": ["jaundice", "dark_urine", "breathlessness"]
    },
    "Dengue": {
        "primary": ["high_fever", "headache", "skin_rash", "muscle_pain", "joint_pain", "pain_behind_the_eyes"],
        "secondary": ["nausea", "vomiting", "fatigue", "loss_of_appetite"],
        "rare": ["abdominal_pain", "breathlessness", "bleeding"]
    },
    "Typhoid": {
        "primary": ["high_fever", "headache", "stomach_pain", "loss_of_appetite", "fatigue"],
        "secondary": ["nausea", "vomiting", "diarrhoea", "constipation", "muscle_pain"],
        "rare": ["skin_rash", "abdominal_pain", "sweating"]
    },
    "Common Cold": {
        "primary": ["continuous_sneezing", "runny_nose", "throat_irritation", "mild_fever", "headache"],
        "secondary": ["chills", "fatigue", "cough", "congestion"],
        "rare": ["muscle_pain", "loss_of_appetite"]
    },
    "Pneumonia": {
        "primary": ["high_fever", "cough", "breathlessness", "chest_pain", "fatigue"],
        "secondary": ["chills", "sweating", "nausea", "vomiting", "headache"],
        "rare": ["muscle_pain", "loss_of_appetite", "abdominal_pain"]
    },
    "Tuberculosis": {
        "primary": ["cough", "weight_loss", "fatigue", "night_sweats", "fever"],
        "secondary": ["chest_pain", "breathlessness", "loss_of_appetite", "chills"],
        "rare": ["blood_in_sputum", "swollen_lymph_nodes", "muscle_pain"]
    },
    "Diabetes": {
        "primary": ["polyuria", "fatigue", "weight_loss", "blurred_and_distorted_vision", "excessive_hunger"],
        "secondary": ["increased_appetite", "fatigue", "skin_rash", "loss_of_balance"],
        "rare": ["nausea", "vomiting", "abdominal_pain"]
    },
    "Hypertension": {
        "primary": ["headache", "dizziness", "chest_pain", "fatigue"],
        "secondary": ["breathlessness", "nausea", "blurred_and_distorted_vision"],
        "rare": ["vomiting", "loss_of_balance", "irregular_sugar_level"]
    },
    "Migraine": {
        "primary": ["headache", "nausea", "blurred_and_distorted_vision", "vomiting"],
        "secondary": ["fatigue", "loss_of_appetite", "dizziness", "excessive_hunger"],
        "rare": ["neck_pain", "loss_of_balance"]
    },
    "Gastroenteritis": {
        "primary": ["vomiting", "diarrhoea", "stomach_pain", "nausea", "dehydration"],
        "secondary": ["high_fever", "fatigue", "loss_of_appetite", "headache"],
        "rare": ["muscle_pain", "chills", "sunken_eyes"]
    },
    "Hepatitis A": {
        "primary": ["jaundice", "fatigue", "nausea", "vomiting", "abdominal_pain", "loss_of_appetite"],
        "secondary": ["dark_urine", "mild_fever", "headache", "joint_pain"],
        "rare": ["itching", "weight_loss", "diarrhoea"]
    },
    "Hepatitis B": {
        "primary": ["jaundice", "fatigue", "abdominal_pain", "dark_urine", "loss_of_appetite"],
        "secondary": ["nausea", "vomiting", "mild_fever", "joint_pain"],
        "rare": ["itching", "weight_loss", "muscle_pain"]
    },
    "Chicken pox": {
        "primary": ["skin_rash", "itching", "high_fever", "fatigue", "loss_of_appetite"],
        "secondary": ["headache", "mild_fever", "muscle_pain", "nausea"],
        "rare": ["vomiting", "abdominal_pain", "breathlessness"]
    },
    "Bronchial Asthma": {
        "primary": ["breathlessness", "cough", "wheezing", "chest_pain", "fatigue"],
        "secondary": ["mild_fever", "throat_irritation", "congestion"],
        "rare": ["nausea", "muscle_pain", "headache"]
    },
    "Heart attack": {
        "primary": ["chest_pain", "breathlessness", "sweating", "nausea", "fatigue"],
        "secondary": ["vomiting", "dizziness", "fast_heart_rate", "anxiety"],
        "rare": ["headache", "loss_of_balance", "weakness_in_limbs"]
    },
    "Allergy": {
        "primary": ["continuous_sneezing", "skin_rash", "itching", "watering_from_eyes", "runny_nose"],
        "secondary": ["chills", "fatigue", "throat_irritation", "congestion"],
        "rare": ["headache", "nausea", "breathlessness"]
    },
    "Urinary tract infection": {
        "primary": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
        "secondary": ["mild_fever", "fatigue", "nausea", "abdominal_pain"],
        "rare": ["vomiting", "chills", "loss_of_appetite"]
    },
    "Jaundice": {
        "primary": ["jaundice", "fatigue", "nausea", "abdominal_pain", "dark_urine"],
        "secondary": ["loss_of_appetite", "vomiting", "mild_fever", "itching"],
        "rare": ["weight_loss", "joint_pain", "diarrhoea"]
    },
    "Fungal infection": {
        "primary": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
        "secondary": ["fatigue", "mild_fever", "loss_of_appetite"],
        "rare": ["nausea", "headache", "muscle_pain"]
    },
    "GERD": {
        "primary": ["stomach_pain", "acidity", "vomiting", "chest_pain", "throat_irritation"],
        "secondary": ["nausea", "loss_of_appetite", "indigestion", "fatigue"],
        "rare": ["headache", "cough", "breathlessness"]
    },
    "Peptic ulcer disease": {
        "primary": ["stomach_pain", "vomiting", "loss_of_appetite", "indigestion", "nausea"],
        "secondary": ["fatigue", "weight_loss", "headache", "acidity"],
        "rare": ["dark_urine", "breathlessness", "chest_pain"]
    },
    "Arthritis": {
        "primary": ["joint_pain", "swelling_joints", "muscle_weakness", "movement_stiffness", "fatigue"],
        "secondary": ["mild_fever", "loss_of_appetite", "weight_loss", "headache"],
        "rare": ["skin_rash", "breathlessness", "chest_pain"]
    },
    "Hypothyroidism": {
        "primary": ["fatigue", "weight_gain", "cold_hands_and_feets", "constipation", "depression"],
        "secondary": ["muscle_weakness", "headache", "loss_of_appetite", "dizziness"],
        "rare": ["nausea", "vomiting", "skin_rash"]
    },
    "Hyperthyroidism": {
        "primary": ["fatigue", "mood_swings", "weight_loss", "fast_heart_rate", "irritability"],
        "secondary": ["sweating", "muscle_weakness", "breathlessness", "headache"],
        "rare": ["nausea", "vomiting", "diarrhoea"]
    },
    "Hypoglycemia": {
        "primary": ["fatigue", "sweating", "headache", "nausea", "anxiety"],
        "secondary": ["dizziness", "vomiting", "loss_of_balance", "irregular_sugar_level"],
        "rare": ["breathlessness", "chest_pain", "muscle_pain"]
    },
}

import numpy as np
import random

# Build symptom list
all_symptoms_set = set()
for disease, data in MEDICAL_DATA.items():
    for cat in ['primary', 'secondary', 'rare']:
        all_symptoms_set.update(data[cat])

# Add more symptoms
extra_symptoms = [
    'polyuria','blurred_and_distorted_vision','excessive_hunger',
    'increased_appetite','irregular_sugar_level','depression',
    'anxiety','mood_swings','restlessness','lethargy',
    'patches_in_throat','high_fever','sunken_eyes',
    'breathlessness','dehydration','indigestion',
    'loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','mild_fever','yellow_urine',
    'yellowing_of_eyes','fluid_overload','swelled_lymph_nodes',
    'malaise','phlegm','throat_irritation','redness_of_eyes',
    'sinus_pressure','runny_nose','congestion','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region',
    'bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','puffy_face_and_eyes',
    'enlarged_thyroid','brittle_nails','swollen_extremeties',
    'extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
    'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck',
    'swelling_joints','movement_stiffness','spinning_movements',
    'loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of_urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching',
    'depression','irritability','muscle_pain','altered_sensorium',
    'red_spots_over_body','belly_pain','abnormal_menstruation',
    'dischromic_patches','watering_from_eyes','increased_appetite',
    'polyuria','family_history','mucoid_sputum','rusty_sputum',
    'lack_of_concentration','visual_disturbances','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking',
    'pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails',
    'blister','red_sore_around_nose','yellow_crust_ooze',
    'nodal_skin_eruptions','dischromic_patches','jaundice',
    'dark_urine','cold_hands_and_feets','night_sweats','wheezing',
    'chest_pain','varicose_veins','irregular_sugar_level'
]

all_symptoms_set.update(extra_symptoms)
all_symptoms = sorted(list(all_symptoms_set))

print(f"Total symptoms: {len(all_symptoms)}")
print(f"Total diseases: {len(MEDICAL_DATA)}")

# Generate dataset
rows = []
for disease, data in MEDICAL_DATA.items():
    primary   = data['primary']
    secondary = data['secondary']
    rare      = data['rare']

    # Generate 200 varied rows per disease
    for i in range(200):
        row = {s: 0 for s in all_symptoms}

        # Always include 2-3 primary symptoms
        n_primary = random.randint(2, min(4, len(primary)))
        selected  = random.sample(primary, n_primary)

        # Add 1-3 secondary symptoms
        n_secondary = random.randint(1, min(3, len(secondary)))
        selected   += random.sample(secondary, n_secondary)

        # Rarely add rare symptoms (30% chance)
        if random.random() < 0.3 and rare:
            selected += random.sample(rare, 1)

        for s in selected:
            if s in row:
                row[s] = 1

        row['prognosis'] = disease
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('data_v3/training.csv', index=False)
print(f"\n✅ Dataset created: {len(df)} rows")
print(f"✅ Diseases: {df['prognosis'].nunique()}")
print(f"✅ Symptoms: {len(all_symptoms)}")
print(df['prognosis'].value_counts().head(5))
