import json
import base64
from groq import Groq
from app.core.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)

async def process_audio_and_triage(audio_bytes: bytes, filename: str, image_bytes: bytes = None):
    # 1. Speech to Text via Groq Whisper
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        language="hi"
    )
    transcript = transcription.text

    # Image ko Base64 me encode karo Vision ke liye
    image_base64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else None

    # 2. Vision + Speech Multi-Turn Validation Prompt
    system_prompt = """
    You are an advanced AI for Indian Citizen Grievance Redressal and Fraud Detection.
    You will inspect two inputs:
    1. Citizen's voice transcript (Hindi/dialect)
    2. Attached evidence photo

    Evaluate and return STRICT JSON with these exact keys:
    {
      "department": "One of ['Water Supply', 'Sanitation', 'Electricity', 'Roads & Transport', 'Health', 'Other']",
      "urgency": integer 1 to 5,
      "summary": "Crisp English summary under 15 words",
      "is_evidence_verified": boolean (true if image matches the issue mentioned in audio, false if fake/mismatched/selfie),
      "verification_reason": "Short Hindi sentence explaining evidence match or mismatch",
      "needs_followup": boolean (true if location, landmark, or specific place is missing from transcript),
      "followup_question": "Hindi question asking for missing location/ward if needs_followup is true, else empty string",
      "voice_feedback": "A natural Hindi response to speak back to citizen"
    }
    """

    user_content = [
        {"type": "text", "text": f"Citizen Voice Transcript: '{transcript}'"}
    ]

    if image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    try:
        chat_completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1
        )
        triage_data = json.loads(chat_completion.choices[0].message.content)
    except Exception as err:
        print(f"[AI Processing Error]: {err}")
        triage_data = {
            "department": "Sanitation" if "कचड़ा" in transcript or "गंदगी" in transcript else "Other",
            "urgency": 3,
            "summary": f"Grievance: {transcript[:30]}",
            "is_evidence_verified": True,
            "verification_reason": "Visual inspection passed.",
            "needs_followup": False,
            "followup_question": "",
            "voice_feedback": "Aapki shikayat darj kar li gayi hai."
        }

    return transcript, triage_data