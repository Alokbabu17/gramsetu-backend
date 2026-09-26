from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ai_service import process_audio_and_triage
from app.services.db_service import upload_file_to_storage, save_grievance

router = APIRouter()

@router.post("/submit-grievance")
async def submit_grievance(
    audio: UploadFile = File(...),
    image: UploadFile = File(...)
):
    try:
        audio_bytes = await audio.read()
        image_bytes = await image.read()

        # Step 1: Upload to Supabase Storage
        audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
        image_url = await upload_file_to_storage(image_bytes, "jpg", "images")

        # Step 2: AI Processing (Whisper + Llama3)
        transcript, triage = await process_audio_and_triage(audio_bytes, audio.filename or "voice.m4a")

        # Step 3: Save to Supabase PostgreSQL
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
            "ticket_id": record["id"],
            "transcript": transcript,
            "triage": triage,
            "audio_url": audio_url,
            "image_url": image_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    return {"status": "live", "engine": "GramSetu FastAPI"}