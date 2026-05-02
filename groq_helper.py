from groq import Groq
import json

LANGUAGES = {
    "English": "English",
    "Arabic (العربية)": "Arabic",
    "Urdu (اردو)": "Urdu",
    "Hindi (हिंदी)": "Hindi",
    "Spanish (Español)": "Spanish",
    "French (Français)": "French",
    "German (Deutsch)": "German",
    "Chinese (中文)": "Chinese",
    "Portuguese (Português)": "Portuguese",
    "Russian (Русский)": "Russian",
    "Bengali (বাংলা)": "Bengali",
    "Indonesian (Bahasa)": "Indonesian",
    "Japanese (日本語)": "Japanese",
    "Turkish (Türkçe)": "Turkish",
    "Korean (한국어)": "Korean",
    "Italian (Italiano)": "Italian",
    "Vietnamese (Tiếng Việt)": "Vietnamese",
    "Malay (Bahasa Melayu)": "Malay",
    "Persian (فارسی)": "Persian",
    "Swahili (Kiswahili)": "Swahili",
    "Filipino (Tagalog)": "Filipino",
    "Polish (Polski)": "Polish",
    "Ukrainian (Українська)": "Ukrainian",
    "Dutch (Nederlands)": "Dutch",
    "Thai (ภาษาไทย)": "Thai",
    "Greek (Ελληνικά)": "Greek",
    "Swedish (Svenska)": "Swedish",
    "Romanian (Română)": "Romanian",
    "Punjabi (ਪੰਜਾਬੀ)": "Punjabi",
    "Amharic (አማርኛ)": "Amharic",
}

MODEL = "llama-3.3-70b-versatile"

def init_groq(api_key):
    return Groq(api_key=api_key)

def smart_chat(client, user_message, collected_symptoms,
               all_symptoms, language, follow_up_count):

    symptoms_str = '\n'.join([f"- {s}" for s in all_symptoms])

    prompt = f"""You are DiagnosBot, a medical AI assistant.

EXACT SYMPTOMS LIST (only pick from this):
{symptoms_str}

User message: "{user_message}"
Already collected symptoms: {collected_symptoms}
Follow-up questions asked: {follow_up_count}

TASK:
1. Extract symptoms from user message matching the list
2. Common mappings:
   - "fever" / "high fever" → high_fever
   - "mild fever" → mild_fever
   - "headache" / "head pain" → headache
   - "body pain" / "body ache" / "muscle ache" → muscle_pain
   - "stomach pain" / "belly pain" → stomach_pain
   - "vomiting" / "vomit" → vomiting
   - "nausea" → nausea
   - "cough" → cough
   - "cold" / "runny nose" → runny_nose
   - "chills" / "shivering" → chills
   - "sweating" → sweating
   - "fatigue" / "tired" / "weakness" → fatigue
   - "diarrhea" / "loose motion" → diarrhoea
   - "joint pain" → joint_pain
   - "chest pain" → chest_pain
   - "back pain" → back_pain
   - "itching" → itching
   - "rash" → skin_rash
   - "breathless" → breathlessness
   - "dizziness" → dizziness
   - "loss of appetite" → loss_of_appetite
   - "weight loss" → weight_loss
   - "sore throat" → throat_irritation
   - "eye redness" → redness_of_eyes
   - "yellow skin" / "jaundice" → yellowish_skin
   - "dark urine" → dark_urine
   - "abdominal pain" → abdominal_pain

Respond ONLY in this exact JSON:
{{
  "extracted_symptoms": ["exact_symptom_from_list"],
  "has_enough": true or false,
  "follow_up": "ONE new question not already answered",
  "is_greeting": true or false
}}

Rules:
- has_enough = true if total symptoms >= 3
- if follow_up_count >= 4, has_enough = true regardless
- is_greeting = true only for hello/thanks
- follow_up must ask about a NEW symptom category not yet discussed
- NEVER ask about duration or severity again if already mentioned
- Ask about: associated symptoms they haven't mentioned yet
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400
    )

    try:
        text  = response.choices[0].message.content.strip()
        text  = text.replace('```json','').replace('```','').strip()
        start = text.find('{')
        end   = text.rfind('}') + 1
        if start >= 0 and end > start:
            text = text[start:end]
        result = json.loads(text)
        valid  = [s for s in result.get('extracted_symptoms',[])
                  if s in all_symptoms]
        result['extracted_symptoms'] = valid
        return result
    except:
        return {
            "extracted_symptoms": [],
            "has_enough": False,
            "follow_up": "Do you have any other symptoms like nausea, chills, or body pain?",
            "is_greeting": False
        }

def get_smart_followup(client, symptoms, language, asked_questions=[]):
    prompt = f"""You are DiagnosBot medical assistant.

Patient symptoms so far: {symptoms}
Questions already asked: {asked_questions}

Ask ONE short follow-up question about a symptom NOT yet discussed.
Focus on symptoms that help narrow down the diagnosis.

Good questions to ask (pick ONE not already asked):
- "Do you have any nausea or vomiting?"
- "Do you have chills or sweating?"
- "Any skin rash or itching?"
- "Do you have stomach pain or loss of appetite?"
- "Any difficulty breathing?"
- "Do you have joint or muscle pain?"
- "Any yellowing of skin or eyes?"

Ask in {language}. ONE question only. Max 20 words.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

def explain_diagnosis(client, disease, confidence,
                      symptoms, precautions, description, language):
    prompt = f"""You are DiagnosBot, warm friendly AI medical assistant.

Diagnosis:
- Disease: {disease}
- Confidence: {confidence}%
- Symptoms: {', '.join(symptoms)}
- Description: {description}
- Precautions: {', '.join(precautions)}

Write warm friendly diagnosis in {language}:
1. Start with empathy
2. State likely condition naturally
3. Mention confidence
4. List precautions warmly
5. End: "Please consult a real doctor for proper medical advice"
6. Max 100 words
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

def general_response(client, user_message, language):
    prompt = f"""You are DiagnosBot, friendly AI medical assistant.
User: "{user_message}"
Respond warmly max 40 words in {language}.
If greeting → greet and ask about symptoms.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=80
    )
    return response.choices[0].message.content.strip()

EMERGENCY_KEYWORDS = [
    "can't breathe","cannot breathe","difficulty breathing",
    "chest pain","left arm pain","heart attack","stroke",
    "unconscious","fainted","seizure","convulsion",
    "severe bleeding","poisoning","overdose","suicidal",
    "can't move","paralyzed","choking",
    "سكتة","نوبة قلبية","لا أتنفس",
    "دل کا دورہ","سانس نہیں",
    "दिल का दौरा","सांस नहीं",
]

EMERGENCY_RESPONSE = {
    "English":  "🚨 **MEDICAL EMERGENCY!**\n\nYour symptoms may be life-threatening.\n**Call emergency services NOW!**\n\n🇦🇪 UAE: **999**\n🇵🇰 Pakistan: **115**\n🇮🇳 India: **108**\n🌍 International: **112**",
    "Arabic":   "🚨 **طوارئ طبية!**\n\nاتصل بالإسعاف فوراً!\n🇦🇪 الإمارات: **999** | 🌍 **112**",
    "Urdu":     "🚨 **طبی ایمرجنسی!**\n\nفوری ایمرجنسی کو کال کریں!\n🇵🇰 **115** | 🌍 **112**",
    "Hindi":    "🚨 **मेडिकल इमरजेंसी!**\n\nतुरंत कॉल करें!\n🇮🇳 **108** | 🌍 **112**",
    "default":  "🚨 **EMERGENCY!**\n\nCall emergency services now!\n🌍 **112**"
}

def check_emergency(message, language):
    msg_lower = message.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw.lower() in msg_lower:
            return EMERGENCY_RESPONSE.get(language, EMERGENCY_RESPONSE["default"])
    return None

def detect_language(client, message):
    prompt = f"""Detect language of: "{message}"
Reply with ONLY the language name in English. Example: Arabic, Urdu, Hindi, French
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10
    )
    detected = response.choices[0].message.content.strip()
    for key, val in LANGUAGES.items():
        if val.lower() == detected.lower():
            return val
    return "English"
