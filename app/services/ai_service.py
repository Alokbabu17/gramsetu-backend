import json
import base64
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
    current_text = transcription.text
    full_transcript = f"{previous_context} | {current_text}".strip(" | ") if previous_context else current_text

    # 2. Strict Guardrail Evaluation Prompt
    system_prompt = """
    You are an AI Civic Grievance & Evidence Inspector for Rural India.
    Your #1 priority is FRAUD DETECTION & AUTHENTICITY.

    STRICT OPERATING RULES:
    1. Check the grievance topic (e.g. Handpump/water, Garbage/cleanliness, Road/pothole, Electricity).
    2. If the user provided photo or description does NOT match real ground civic infrastructure (e.g. laptop screen, coding, indoor room, random object, selfie), you MUST:
       - "is_evidence_verified": false
       - "needs_followup": false
       - "verification_reason": "Evidence photo does not show civic physical issue matching the problem."
       - "voice_feedback": "Dhyan dein, aapki photo aapki samasya se match nahi ho rahi hai. Kripya asli samasya ki photo khinchein."
       - NEVER set "needs_followup": true when evidence is unverified or mismatched. REJECT IMMEDIATELY.

    3. ONLY IF evidence looks authentic:
       - Check if location/ward/landmark is mentioned in the full transcript.
       - If location is missing:
         "needs_followup": true
         "followup_question": "Kripya batayein ye samasya kaunse ward me ya kiske ghar ke paas hai?"
         "voice_feedback": "Kripya batayein ye samasya kaunse ward me ya kiske ghar ke paas hai?"
       - If location is present:
         "needs_followup": false
         "voice_feedback": "Aapki shikayat darj kar li gayi hai."

    Output STRICT JSON:
    {
      "department": "One of ['Water Supply', 'Sanitation', 'Electricity', 'Roads & Transport', 'Health', 'Other']",
      "urgency": 1 to 5,
      "summary": "Crisp summary under 15 words",
      "is_evidence_verified": true or false,
      "verification_reason": "Reason in Hindi",
      "needs_followup": true or false,
      "followup_question": "Question or empty",
      "voice_feedback": "Hindi voice feedback"
    }
    """

    model_name = get_active_model()
    try:
        chat_completion = groq_client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Citizen Grievance Full Transcript: '{full_transcript}'"}
            ],
            temperature=0.1
        )
        triage_data = json.loads(chat_completion.choices[0].message.content)
    except Exception as err:
        print(f"[AI Evaluation Error]: {err}")
        triage_data = {
            "department": "Other",
            "urgency": 3,
            "summary": full_transcript[:25],
            "is_evidence_verified": False,
            "verification_reason": "Jaanch me asafal. Kripya punah photo lein.",
            "needs_followup": False,
            "followup_question": "",
            "voice_feedback": "Photo ki jaanch nahi ho saki. Sahi photo punah khinchein."
        }

    return full_transcript, triage_data