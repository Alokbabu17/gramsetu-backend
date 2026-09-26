import json
from groq import Groq
from app.core.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)

# Priority order of current active Groq models
CANDIDATE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

async def process_audio_and_triage(audio_bytes: bytes, filename: str):
    # 1. Speech to Text
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        language="hi"
    )
    transcript = transcription.text

    # 2. Triage Extraction
    system_prompt = """
    You are an AI for Indian Citizen Grievance Redressal.
    Given a citizen's complaint transcription, extract:
    1. department: One of ["Water Supply", "Sanitation", "Electricity", "Roads & Transport", "Health", "Other"]
    2. urgency: Integer from 1 (minor issue) to 5 (critical/emergency)
    3. summary: A crisp English summary of the issue (max 20 words)

    Output STRICTLY valid JSON with no markdown wrapping:
    {
      "department": "...",
      "urgency": 1,
      "summary": "..."
    }
    """

    triage_data = None
    for model_name in CANDIDATE_MODELS:
        try:
            chat_completion = groq_client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Citizen Complaint Text: {transcript}"}
                ],
                temperature=0.1
            )
            triage_data = json.loads(chat_completion.choices[0].message.content)
            print(f"[LLM Success with model]: {model_name}")
            break
        except Exception as e:
            print(f"[Model {model_name} failed]: {e}")
            continue

    if not triage_data:
        triage_data = {
            "department": "Sanitation" if "कचड़ा" in transcript or "गंदगी" in transcript else "Other",
            "urgency": 4 if "कचड़ा" in transcript else 3,
            "summary": f"Citizen grievance regarding: {transcript[:40]}"
        }

    return transcript, triage_data