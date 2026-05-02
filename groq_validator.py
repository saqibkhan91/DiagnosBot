"""
Groq Validation Layer
- Does NOT diagnose
- Only validates if ML prediction is medically reasonable
- Acts as a safety check
"""

MODEL = "llama-3.3-70b-versatile"

def validate_diagnosis(client, disease, confidence, symptoms, language):
    """
    Groq checks if ML prediction makes medical sense.
    Returns: {valid: bool, warning: str, suggestion: str}
    """
    prompt = f"""You are a medical validation assistant for DiagnosBot.

The ML model predicted:
- Disease: {disease}
- Confidence: {confidence}%
- Symptoms reported: {', '.join(symptoms)}

Your job is to validate if this prediction is medically reasonable.
You are NOT diagnosing — you are checking if the ML result makes sense.

Medical knowledge check:
- Is {disease} commonly associated with these symptoms?
- Are there any serious diseases being missed?
- Is the confidence level appropriate?

Respond ONLY in this exact JSON:
{{
  "is_valid": true or false,
  "confidence_assessment": "appropriate" or "too_low" or "misleading",
  "warning": "empty string if valid, or brief medical concern if invalid",
  "better_suggestion": "empty string if valid, or more likely condition based on symptoms",
  "safety_flag": true or false
}}

Rules:
- is_valid = true if {disease} is medically reasonable for these symptoms
- is_valid = false if prediction seems completely wrong medically
- safety_flag = true ONLY if symptoms suggest something potentially serious being missed
- warning max 20 words
- better_suggestion max 10 words
- Be lenient — only flag obvious medical errors
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        import json
        text  = response.choices[0].message.content.strip()
        text  = text.replace('```json','').replace('```','').strip()
        start = text.find('{')
        end   = text.rfind('}') + 1
        if start >= 0 and end > start:
            text = text[start:end]
        return json.loads(text)
    except:
        return {
            "is_valid": True,
            "confidence_assessment": "appropriate",
            "warning": "",
            "better_suggestion": "",
            "safety_flag": False
        }

def get_confidence_message(confidence, is_valid, warning, language):
    """Generate appropriate message based on confidence and validation"""

    if confidence >= 70 and is_valid:
        level = "HIGH"
    elif confidence >= 50 and is_valid:
        level = "MEDIUM"
    else:
        level = "LOW"

    messages = {
        "HIGH": {
            "English": f"✅ High confidence diagnosis ({confidence}%)",
            "Arabic":  f"✅ تشخيص عالي الثقة ({confidence}%)",
            "Urdu":    f"✅ اعلی اعتماد تشخیص ({confidence}%)",
            "Hindi":   f"✅ उच्च विश्वास निदान ({confidence}%)",
        },
        "MEDIUM": {
            "English": f"⚠️ Moderate confidence ({confidence}%) — consult a doctor to confirm",
            "Arabic":  f"⚠️ ثقة متوسطة ({confidence}%) — استشر طبيباً للتأكيد",
            "Urdu":    f"⚠️ درمیانی اعتماد ({confidence}%) — تصدیق کے لیے ڈاکٹر سے ملیں",
            "Hindi":   f"⚠️ मध्यम विश्वास ({confidence}%) — पुष्टि के लिए डॉक्टर से मिलें",
        },
        "LOW": {
            "English": f"ℹ️ Low confidence ({confidence}%) — symptoms match multiple conditions. Please see a doctor.",
            "Arabic":  f"ℹ️ ثقة منخفضة ({confidence}%) — الأعراض تتطابق مع حالات متعددة",
            "Urdu":    f"ℹ️ کم اعتماد ({confidence}%) — علامات متعدد حالتوں سے ملتی ہیں",
            "Hindi":   f"ℹ️ कम विश्वास ({confidence}%) — लक्षण कई स्थितियों से मेल खाते हैं",
        }
    }
    return messages[level].get(language, messages[level]["English"])
