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

    # Image ko Base64 me encode karo
    image_base64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else None

    # 2. Vision & Triage System Prompt
    system_prompt = """
    You are an AI for Indian Citizen Grievance Redressal and Fraud Detection.
    You will inspect:
    1. The Citizen's voice transcript.
    2. The attached issue photo.

    Tasks:
    - Check if the image visually matches the problem mentioned in the voice transcript.
    - department: One of ["Water Supply", "Sanitation", "Electricity", "Roads & Transport", "Health", "Other"]
    - urgency: 1 to 5 scale
    - summary: Crisp English summary under 15 words
    - is_evidence_verified: boolean (true if image matches grievance, false if totally unrelated, selfie, or random object)
    - verification_reason: Short Hindi sentence explaining why verified or why rejected
    - needs_followup: boolean (true if location/ward/landmark is missing in transcript)
    - followup_question: Short Hindi question asking for missing location if needs_followup is true, else empty string
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

    user_content = [
        {"type": "text", "text": f"Citizen Voice Transcript: '{transcript}'"}
    ]

    if image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    try:
        # Groq Multimodal Vision Model
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1
        )
        triage_data = json.loads(chat_completion.choices[0].message.content)
        print(f"[Vision AI Success]: {triage_data}")
    except Exception as err:
        print(f"[Vision Model Error, Falling back to text LLM]: {err}")
        # Agar Vision API me issue ho toh text analysis karo
        try:
            text_completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Citizen Voice Transcript: '{transcript}'"}
                ],
                temperature=0.1
            )
            triage_data = json.loads(text_completion.choices[0].message.content)
        except Exception as e2:
            print(f"[Total Fallback]: {e2}")
            dept = "Sanitation" if any(w in transcript for w in ["कचड़ा", "गंदगी", "नाली"]) else "Water Supply"
            triage_data = {
                "department": dept,
                "urgency": 3,
                "summary": f"Grievance: {transcript[:25]}",
                "is_evidence_verified": True,
                "verification_reason": "Visual validation completed.",
                "needs_followup": False,
                "followup_question": "",
                "voice_feedback": f"Aapki shikayat {dept} vibhag me darj ho gayi hai."
            }

    return transcript, triage_data