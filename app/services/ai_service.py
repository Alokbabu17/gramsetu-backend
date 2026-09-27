import json
from groq import Groq
import google.generativeai as genai
from app.core.config import settings

# 1. Groq Client (Fast STT)
groq_client = Groq(api_key=settings.GROQ_API_KEY)

# 2. Gemini Configuration
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

def get_gemini_vision_model():
    """Google recommended active Flash Vision model resolve karta hai."""
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Check active models suggested by API error
        for target in ["models/gemini-3.8-flash", "gemini-3.8-flash", "models/gemini-2.0-flash", "gemini-2.0-flash"]:
            if any(target in m for m in available):
                return genai.GenerativeModel(target)
        for m in available:
            if "flash" in m:
                return genai.GenerativeModel(m)
    except Exception as e:
        print(f"[Gemini ListModels Fallback]: {e}")
    # Google recommended default from log
    return genai.GenerativeModel("gemini-3.8-flash")

async def process_audio_and_triage(
    audio_bytes: bytes,
    filename: str,
    image_bytes: bytes = None,
    previous_context: str = ""
):
    # Step A: Whisper Audio Transcription via Groq
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        language="hi"
    )
    current_transcript = transcription.text.strip()
    full_transcript = f"{previous_context} | {current_transcript}".strip(" | ") if previous_context else current_transcript
    print(f"[Audio Transcript]: {full_transcript}")

    # Step B: Gemini Multimodal Vision Verification
    if settings.GEMINI_API_KEY and image_bytes:
        try:
            model = get_gemini_vision_model()

            prompt = f"""
            You are an AI Civic Redressal Inspector and Fraud Detection Officer for Rural India (GramSetu).
            Analyze both:
            1. The citizen's voice grievance: "{full_transcript}"
            2. The attached evidence image.

            Tasks:
            1. What is physically visible in this image? (e.g. handpump, water tap, garbage, potholes, road, laptop screen, selfie).
            2. Cross-Verification: Does the image show the physical civic problem mentioned in the transcript?
               - If audio says handpump/water and image shows handpump -> is_evidence_verified MUST BE true.
               - If audio says handpump/water and image shows garbage/laptop/selfie/indoor room -> is_evidence_verified MUST BE false.
               - If audio says garbage/kachra and image shows garbage/dump -> is_evidence_verified MUST BE true.
            3. Department: One of ["Water Supply", "Sanitation", "Electricity", "Roads & Transport", "Health", "Other"]
            4. Urgency: integer 1 to 5.
            5. Summary: Crisp English summary under 12 words.
            6. Verification Reason: 1 clear Hindi sentence explaining verification result.
            7. Voice Feedback: Natural Hindi response for the citizen.

            Output STRICT JSON ONLY (no markdown blocks, no backticks, just raw json):
            {{
              "department": "Water Supply",
              "urgency": 3,
              "summary": "Broken handpump reported",
              "is_evidence_verified": true,
              "verification_reason": "Tasveer me handpump dikh raha hai jo aawaz se match hota hai.",
              "needs_followup": false,
              "followup_question": "",
              "voice_feedback": "Aapki shikayat Water Supply vibhag me darj ho chuki hai."
            }}
            """

            image_part = {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }

            response = model.generate_content([prompt, image_part])
            raw_text = response.text.strip()

            # Clean markdown formatting if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            triage_data = json.loads(raw_text.strip())
            print(f"[Gemini Vision Verification Success]: {triage_data}")
            return full_transcript, triage_data

        except Exception as e:
            print(f"[Gemini Vision Error]: {e}")

    # Fallback department mapping based on audio keywords
    dept = "Water Supply" if any(w in full_transcript for w in ["पानी", "नल", "हैंड पंप", "हैंडपंप", "जल"]) else "Sanitation"
    triage_data = {
        "department": dept,
        "urgency": 3,
        "summary": full_transcript[:30],
        "is_evidence_verified": True,
        "verification_reason": "Shikayat darj ki gayi.",
        "needs_followup": False,
        "followup_question": "",
        "voice_feedback": f"Aapki shikayat {dept} vibhag me darj ho gayi hai."
    }
    return full_transcript, triage_data