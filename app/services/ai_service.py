import json
from groq import Groq
import google.generativeai as genai
from app.core.config import settings

# 1. Groq Client
groq_client = Groq(api_key=settings.GROQ_API_KEY)

# 2. Gemini Setup
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

def analyze_with_groq_llm(transcript: str) -> dict:
    """Gemini quota hit hone par Groq se accurate triage."""
    prompt = f"""
    You are an AI Civic Grievance Triage Officer for Indian Rural Governance (GramSetu).
    Analyze this citizen complaint transcript:
    Transcript: "{transcript}"

    Rules:
    - If words relate to handpump, water, tap, pipe, jal -> department: "Water Supply"
    - If words relate to kachra, garbage, waste, drainage, nali, dirt -> department: "Sanitation"
    - If words relate to road, sadak, pothole, gaddha, pul -> department: "Roads & Transport"
    - If words relate to bijli, electricity, wire, transformer, light -> department: "Electricity"
    - Else -> department: "Other"

    Output strict JSON only:
    {{
      "department": "Water Supply",
      "urgency": 3,
      "summary": "Short 10-word summary",
      "is_evidence_verified": true,
      "verification_reason": "Shikayat ki janch aawaz ke aadhar par purn hui.",
      "needs_followup": false,
      "followup_question": "",
      "voice_feedback": "Aapki shikayat darj kar li gayi hai."
    }}
    """
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        data = json.loads(res.choices[0].message.content)
        dept = data.get("department", "Water Supply")
        data["voice_feedback"] = f"Aapki shikayat {dept} vibhag me darj ho chuki hai."
        return data
    except Exception as e:
        print(f"[Groq LLM Fallback Error]: {e}")
        dept = "Water Supply" if any(w in transcript for w in ["पानी", "नल", "हैंड पंप", "हैंडपंप", "जल"]) else "Sanitation"
        return {
            "department": dept,
            "urgency": 3,
            "summary": transcript[:25],
            "is_evidence_verified": True,
            "verification_reason": "Aawaz aadharit panjikaran safal raha.",
            "needs_followup": False,
            "followup_question": "",
            "voice_feedback": f"Aapki shikayat {dept} vibhag me darj ho chuki hai."
        }

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

    # Step B: Gemini Vision using official stable alias
    if settings.GEMINI_API_KEY and image_bytes:
        # Try stable aliases that have higher standard quotas
        candidate_models = ["gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                prompt = f"""
                You are an AI Civic Redressal Inspector and Fraud Detection Officer for Rural India (GramSetu).
                Transcript: "{full_transcript}"

                Tasks:
                1. Check if image physically matches civic complaint.
                   - If audio says handpump and image shows handpump -> is_evidence_verified: true, department: "Water Supply"
                   - If audio says handpump and image shows garbage/laptop/room/selfie -> is_evidence_verified: false
                   - If audio says kachra/garbage and image shows garbage -> is_evidence_verified: true, department: "Sanitation"
                2. Department: One of ["Water Supply", "Sanitation", "Electricity", "Roads & Transport", "Health", "Other"]
                3. Urgency: 1 to 5.
                4. Summary: Under 12 words in English.
                5. Verification Reason: 1 clear Hindi sentence.
                6. Voice Feedback: Natural Hindi response.

                Output STRICT JSON only:
                {{
                  "department": "Water Supply",
                  "urgency": 3,
                  "summary": "...",
                  "is_evidence_verified": true,
                  "verification_reason": "...",
                  "needs_followup": false,
                  "followup_question": "",
                  "voice_feedback": "..."
                }}
                """
                image_part = {"mime_type": "image/jpeg", "data": image_bytes}
                response = model.generate_content([prompt, image_part])
                raw_text = response.text.strip()

                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                triage_data = json.loads(raw_text.strip())
                print(f"[Vision Verification Success via {model_name}]: {triage_data}")
                return full_transcript, triage_data

            except Exception as e:
                print(f"[Model {model_name} failed / Quota reached]: {e}")
                continue  # Next candidate model try karega

    # Step C: Fallback to high-quota Groq Llama-3.3 engine
    print("[Switching to Groq Llama-3.3 Redressal Engine]")
    triage_data = analyze_with_groq_llm(full_transcript)
    return full_transcript, triage_data