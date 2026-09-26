import traceback
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
        print("[1] Reading uploaded files...")
        audio_bytes = await audio.read()
        image_bytes = await image.read()

        print("[2] Uploading files to Supabase...")
        try:
            audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
            image_url = await upload_file_to_storage(image_bytes, "jpg", "images")
        except Exception as storage_err:
            print(f"[ERROR in Supabase Storage]: {storage_err}")
            traceback.print_exc()
            # Storage fail hone par dummy URL dekar aage AI chalne do
            audio_url = "https://placeholder.url/audio.m4a"
            image_url = "https://placeholder.url/image.jpg"

        print("[3] Calling Groq Whisper & Llama-3...")
        transcript, triage = await process_audio_and_triage(audio_bytes, audio.filename or "voice.m4a")

        print("[4] Saving record to Supabase DB...")
        try:
            record = await save_grievance(
                transcript=transcript,
                department=triage.get("department", "Other"),
                urgency=triage.get("urgency", 3),
                summary=triage.get("summary", ""),
                audio_url=audio_url,
                image_url=image_url
            )
            ticket_id = record["id"]
        except Exception as db_err:
            print(f"[ERROR in Supabase DB Insert]: {db_err}")
            traceback.print_exc()
            ticket_id = "LOCAL-TX-1001"

        return {
            "success": True,
            "ticket_id": ticket_id,
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
    return {"status": "live", "engine": "GramSetu FastAPI"}