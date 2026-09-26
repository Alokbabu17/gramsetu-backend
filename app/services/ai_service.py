import json
import base64
from groq import Groq
from app.core.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)

def get_active_groq_model(prefer_vision: bool = True) -> str:
    """Groq API se active models ki list dynamic fetch karta hai."""
    try:
        available_models = [m.id for m in groq_client.models.list().data]
        if prefer_vision:
            # Check for any active vision model
            for m in available_models:
                if "vision" in m.lower():
                    return m
        # Agar vision na ho toh text models priority
        for preferred in ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            if preferred in available_models:
                return preferred
        return available_models[0]
    except Exception as e:
        print(f"[Model List Fetch Failed]: {e}")
        return "openai/gpt-oss-120b"

async def process_audio_and_triage(audio_bytes: bytes, filename: str, image_bytes: bytes = None):
    # 1. Speech to Text via Groq Whisper
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        language="hi"
    )
    transcript = transcription.text

    system_prompt = """
    You are an AI for Indian Citizen Grievance Redressal and Fraud Detection.
    You will inspect the citizen's complaint details.

    Tasks:
    - department: One of ["Water Supply", "Sanitation", "Electricity", "Roads & Transport", "Health", "Other"]
    - urgency: 1 to 5 scale
    - summary: Crisp English summary under 15 words
    - is_evidence_verified: boolean (true if complaint represents genuine civic issue, false if random spam)
    - verification_reason: Short Hindi sentence explaining verification result
    - needs_followup: boolean (true if ward or specific location is completely missing)
    - followup_question: Short Hindi question asking for location if needs_followup is true, else empty string
    - voice_feedback: Natural Hindi response for citizen

    Output STRICT JSON only:
    {
      "department": "...",
      "urgency": 1,
      "summary": "...",
      "is_evidence_verified": true,
      "verification_reason": "...",
      "needs_followup": false,
      "followup_question": "",
      "voice_feedback": "..."
    }
    """

    # Model select karo
    target_model = get_active_groq_model(prefer_vision=False)
    print(f"[Using Groq Model]: {target_model}")

    # Text based analysis using active Groq LLM
    try:
        chat_completion = groq_client.chat.completions.create(
            model=target_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Citizen Voice Transcript: '{transcript}'"}
            ],
            temperature=0.1
        )
        triage_data = json.loads(chat_completion.choices[0].message.content)
    except Exception as err:
        print(f"[Groq LLM Error]: {err}")
        dept = "Water Supply" if any(w in transcript for w in ["नल", "हैंड पंप", "पानी"]) else "Sanitation"
        triage_data = {
            "department": dept,
            "urgency": 4 if "खराब" in transcript or "कचड़ा" in transcript else 3,
            "summary": f"Civic Grievance: {transcript[:25]}",
            "is_evidence_verified": True,
            "verification_reason": "Shikayat jaanch ke liye sweekar ki gayi.",
            "needs_followup": False,
            "followup_question": "",
            "voice_feedback": f"Aapki shikayat {dept} vibhag me darj kar li gayi hai."
        }

    return transcript, triage_data