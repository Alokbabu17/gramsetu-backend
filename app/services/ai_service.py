import json
from groq import Groq
from app.core.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)

async def process_audio_and_triage(audio_bytes: bytes, filename: str):
    # 1. Speech to Text via Groq Whisper
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        language="hi"
    )
    transcript = transcription.text

    # 2. Triage & Extraction via Groq Playground Model
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

    chat_completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Citizen Complaint Text: {transcript}"}
        ],
        temperature=0.1
    )

    triage_data = json.loads(chat_completion.choices[0].message.content)
    return transcript, triage_data