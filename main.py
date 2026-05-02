from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib, os
from groq_helper import (
    init_groq, smart_chat, explain_diagnosis,
    general_response, get_smart_followup, LANGUAGES,
    check_emergency, detect_language
)
from groq_validator import validate_diagnosis, get_confidence_message
from db_helper import init_db, save_session, save_diagnosis, save_feedback
from pdf_helper import generate_pdf

app = FastAPI(title="DiagnosBot v4.0 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load v3 Model ──────────────────────────────────────────
print("🔄 Loading v3 medically accurate model...")
model        = joblib.load('model_v3/model.pkl')
le           = joblib.load('model_v3/label_encoder.pkl')
ALL_SYMPTOMS = joblib.load('model_v3/symptoms.pkl')
print(f"✅ v3 Model loaded — {len(ALL_SYMPTOMS)} symptoms, {len(le.classes_)} diseases")

desc_df = pd.read_csv('data/symptom_description.csv')
prec_df = pd.read_csv('data/symptom_precaution.csv')
init_db()
sessions = {}

def get_description(disease):
    row = desc_df[desc_df['Disease'].str.lower() == disease.lower()]
    return row['Description'].values[0] if not row.empty else ""

def get_precautions(disease):
    row = prec_df[prec_df['Disease'].str.lower() == disease.lower()]
    if row.empty:
        return []
    return [row[f'Precaution_{i}'].values[0] for i in range(1, 5)
            if pd.notna(row[f'Precaution_{i}'].values[0])]

def predict_disease(selected):
    vec = np.zeros(len(ALL_SYMPTOMS))
    for s in selected:
        if s in ALL_SYMPTOMS:
            vec[ALL_SYMPTOMS.index(s)] = 1
    proba    = model.predict_proba([vec])[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    return [(le.inverse_transform([i])[0], round(proba[i] * 100, 1))
            for i in top3_idx]

class InitRequest(BaseModel):
    session_id: str
    api_key:    str
    language:   str
    consent:    bool

class ChatRequest(BaseModel):
    session_id: str
    message:    str

class FeedbackRequest(BaseModel):
    session_id: str
    helpful:    bool
    rating:     int

class PDFRequest(BaseModel):
    session_id:  str
    disease:     str
    confidence:  float
    symptoms:    list
    precautions: list
    description: str
    language:    str

class DetectLangRequest(BaseModel):
    session_id: str
    message:    str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/languages")
def get_languages():
    return {"languages": list(LANGUAGES.keys())}

@app.post("/init")
def init_session(req: InitRequest):
    try:
        client = init_groq(req.api_key)
        sessions[req.session_id] = {
            "client":          client,
            "language":        req.language,
            "consent":         req.consent,
            "symptoms":        [],
            "follow_ups":      0,
            "asked_questions": [],
            "diagnosed":       False,
        }
        save_session(req.session_id, req.language,
                     consent=1 if req.consent else 0)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        return {"status": "error", "message": "Session not found. Please reconnect."}

    try:
        client   = session["client"]
        language = session["language"]

        # Extract symptoms
        result = smart_chat(
            client, req.message,
            session["symptoms"],
            ALL_SYMPTOMS, language,
            session["follow_ups"]
        )

        for s in result.get("extracted_symptoms", []):
            if s not in session["symptoms"]:
                session["symptoms"].append(s)

        collected = session["symptoms"]

        if result.get("is_greeting"):
            return {
                "status":  "general",
                "message": general_response(client, req.message, language)
            }

        MIN_SYMPTOMS = 3

        if len(collected) >= MIN_SYMPTOMS:
            top3              = predict_disease(collected)
            disease, confidence = top3[0]

            # ── Groq Validation Layer ──────────────────────
            validation = validate_diagnosis(
                client, disease, confidence, collected, language
            )

            # If Groq says invalid and confidence low → ask more
            if not validation['is_valid'] and confidence < 50 and session["follow_ups"] < 4:
                session["follow_ups"] += 1
                followup = get_smart_followup(
                    client, collected, language,
                    session["asked_questions"]
                )
                session["asked_questions"].append(followup)

                hint = ""
                if validation.get('better_suggestion'):
                    hint = f" (I need more info to rule out {validation['better_suggestion']})"

                return {
                    "status":   "followup",
                    "message":  followup + hint,
                    "symptoms": collected
                }

            # Get disease info
            desc  = get_description(disease)
            precs = get_precautions(disease)

            # Save to DB
            if session["consent"]:
                save_diagnosis(
                    req.session_id, req.message,
                    collected, disease,
                    confidence, session["follow_ups"]
                )

            # Get Groq explanation
            explanation = explain_diagnosis(
                client, disease, confidence,
                collected, precs, desc, language
            )

            # Add confidence message
            conf_msg = get_confidence_message(
                confidence,
                validation['is_valid'],
                validation.get('warning', ''),
                language
            )

            # Add safety warning if needed
            safety_warning = ""
            if validation.get('safety_flag') and validation.get('warning'):
                safety_warning = f"\n\n⚠️ **Medical Note:** {validation['warning']}"

            session["diagnosed"] = True

            return {
                "status":        "diagnosis",
                "message":       explanation + safety_warning,
                "confidence_msg": conf_msg,
                "top3":          [{"disease": d, "confidence": c} for d, c in top3],
                "symptoms":      collected,
                "disease":       disease,
                "confidence":    confidence,
                "precautions":   precs,
                "validation":    validation,
            }

        else:
            session["follow_ups"] += 1

            if session["follow_ups"] >= 5:
                if collected:
                    top3              = predict_disease(collected)
                    disease, confidence = top3[0]
                    desc  = get_description(disease)
                    precs = get_precautions(disease)
                    explanation = explain_diagnosis(
                        client, disease, confidence,
                        collected, precs, desc, language
                    )
                    session["diagnosed"] = True
                    return {
                        "status":      "diagnosis",
                        "message":     explanation,
                        "confidence_msg": get_confidence_message(confidence, True, "", language),
                        "top3":        [{"disease": d, "confidence": c} for d, c in top3],
                        "symptoms":    collected,
                        "disease":     disease,
                        "confidence":  confidence,
                        "precautions": precs,
                        "validation":  {"is_valid": True, "safety_flag": False},
                    }
                else:
                    return {
                        "status":  "general",
                        "message": general_response(client, req.message, language)
                    }

            followup = get_smart_followup(
                client, collected, language,
                session["asked_questions"]
            )
            session["asked_questions"].append(followup)
            return {
                "status":   "followup",
                "message":  followup,
                "symptoms": collected
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    save_feedback(req.session_id, 1 if req.helpful else 0, req.rating)
    return {"status": "ok"}

@app.post("/generate_pdf")
def generate_pdf_report(req: PDFRequest):
    try:
        filename = generate_pdf(
            req.disease, req.confidence, req.symptoms,
            req.precautions, req.description,
            req.language, req.session_id
        )
        return FileResponse(
            filename,
            media_type='application/pdf',
            headers={"Content-Disposition": "attachment; filename=DiagnosBot_Report.pdf"}
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/detect_language")
def detect_lang(req: DetectLangRequest):
    session = sessions.get(req.session_id)
    if not session:
        return {"language": "English"}
    detected = detect_language(session["client"], req.message)
    return {"language": detected}

@app.post("/check_emergency")
def emergency_check(req: ChatRequest):
    session   = sessions.get(req.session_id)
    lang      = session["language"] if session else "English"
    emergency = check_emergency(req.message, lang)
    if emergency:
        return {"is_emergency": True, "message": emergency}
    return {"is_emergency": False}

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
