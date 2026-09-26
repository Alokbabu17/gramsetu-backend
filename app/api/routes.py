import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.ai_service import process_audio_and_triage
from app.services.db_service import upload_file_to_storage, save_grievance

router = APIRouter()

@router.post("/submit-grievance")
async def submit_grievance(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    user_name: str = Form("Nagrik"),
    user_phone: str = Form(""),
    user_ward: str = Form("Ward 1")
):
    try:
        print(f"[1] Receiving grievance from {user_name} ({user_phone}, {user_ward})...")
        audio_bytes = await audio.read()
        image_bytes = await image.read()

        print("[2] Uploading evidence files...")
        audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
        image_url = await upload_file_to_storage(image_bytes, "jpg", "images")

        print("[3] AI Multi-modal Vision & Speech Triage...")
        transcript, triage = await process_audio_and_triage(
            audio_bytes=audio_bytes,
            filename=audio.filename or "voice.m4a",
            image_bytes=image_bytes
        )

        # Agar photo fake/mismatched hai toh reject alert
        if not triage.get("is_evidence_verified", True):
            reason = triage.get("verification_reason", "Photo aapki samasya se match nahi ho rahi hai.")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason}. Kripya sahi photo khinchein.",
                "transcript": transcript,
                "triage": triage
            }

        # Agar location missing hai toh follow-up question
        if triage.get("needs_followup", False):
            question = triage.get("followup_question", "Kripya batayein ye samasya kis jagah par hai?")
            return {
                "success": False,
                "status": "NEEDS_FOLLOWUP",
                "voice_feedback": question,
                "transcript": transcript,
                "triage": triage
            }

        print("[4] Saving verified ticket to Supabase...")
        record = await save_grievance(
            transcript=transcript,
            department=triage.get("department", "Other"),
            urgency=triage.get("urgency", 3),
            summary=triage.get("summary", ""),
            audio_url=audio_url,
            image_url=image_url
        )

        return {
            "success": True,
            "status": "FILED",
            "ticket_id": record["id"],
            "voice_feedback": f"Aapki shikayat {triage.get('department')} vibhag me darj ho chuki hai.",
            "transcript": transcript,
            "triage": triage,
            "audio_url": audio_url,
            "image_url": image_url
        }
    except Exception as e:
        print(f"[CRITICAL ERROR]: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    return {"status": "live", "engine": "GramSetu FastAPI Multi-Turn"}