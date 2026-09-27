import json
from groq import Groq
from app.core.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)

def get_active_model() -> str:
    try:
        models = [m.id for m in groq_client.models.list().data]
        for preferred in ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]:
            if preferred in models:
                return preferred
        return models[0]
    except Exception:
        return "llama-3.3-70b-versatile"

async def process_audio_and_triage(
    audio_bytes: bytes,
    filename: str,
    image_bytes: bytes = None,
    previous_context: str = ""
):
    # 1. Groq Whisper STT
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        language="hi"
    )
    current_text = transcription.text.strip()
    full_transcript = f"{previous_context} | {current_text}".strip(" | ") if previous_context else current_text

    # 2. Sensible Triage Prompt
    system_prompt = """
    You are an AI Civic Redressal Inspector for Indian rural governance (GramSetu).
    You are reviewing a citizen's complaint transcript.

    RULES:
    1. If the transcript discusses legitimate civic issues (Garbage, Waste, Handpump, Water, Broken Road, Drainage, Streetlight, Electricity, Health):
       - "is_evidence_verified": true
       - "verification_reason": "Nagrik ki shikayat civic mudde se sambandhit hai."
       - Classify proper department: ["Sanitation", "Water Supply", "Roads & Transport", "Electricity", "Health", "Other"]
       - "urgency": 1 to 5
       - "voice_feedback": "Aapki shikayat safalta-purvak darj kar li gayi hai."

    2. ONLY mark "is_evidence_verified": false IF:
       - The transcript is completely gibberish, empty, random personal talk, or intentional abusive spam unrelated to any village issue.
       - If rejected:
         "verification_reason": "Shikayat me gaon ki kisi samasya ka vivaran nahi hai."
         "voice_feedback": "Kripya gaon ki kisi mukhya samasya ke baare me saaf aawaaz me batayein."

    3. Needs Followup:
       - Always set "needs_followup": false for now to avoid blocking genuine rural complaints.

    Output STRICT JSON only:
    {
      "department": "Sanitation / Water Supply / etc",
      "urgency": 3,
      "summary": "Crisp summary under 12 words",
      "is_evidence_verified": true,
      "verification_reason": "...",
      "needs_followup": false,
      "followup_question": "",
      "voice_feedback": "..."
    }
    """

    model_name = get_active_model()
    try:
        chat_completion = groq_client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Citizen Complaint Transcript: '{full_transcript}'"}
            ],
            temperature=0.1
        )
        triage_data = json.loads(chat_completion.choices[0].message.content)
    except Exception as err:
        print(f"[AI Evaluation Error]: {err}")
        # Rule based sensible fallback
        dept = "Sanitation"
        if any(w in full_transcript for w in ["पानी", "नल", "हैंड पंप", "जल"]):
            dept = "Water Supply"
        elif any(w in full_transcript for w in ["सड़क", "रास्ता", "गड्ढा"]):
            dept = "Roads & Transport"
        elif any(w in full_transcript for w in ["बिजली", "लाइट", "तार"]):
            dept = "Electricity"

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