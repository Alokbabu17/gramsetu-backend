import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.services.ai_service import process_audio_and_triage
from app.services.db_service import upload_file_to_storage, save_grievance, supabase

router = APIRouter()

class CitizenLoginRequest(BaseModel):
    phone: str
    name: str
    state: str
    district: str
    ward: str
    language: str = "hi"

@router.post("/citizen-login")
def citizen_login(req: CitizenLoginRequest):
    try:
        # Check if user already exists
        res = supabase.table("citizens").select("*").eq("phone", req.phone).execute()
        if res.data and len(res.data) > 0:
            # Update user info if needed
            supabase.table("citizens").update({
                "name": req.name,
                "state": req.state,
                "district": req.district,
                "ward": req.ward,
                "preferred_language": req.language
            }).eq("phone", req.phone).execute()
        else:
            # Insert new citizen
            supabase.table("citizens").insert({
                "phone": req.phone,
                "name": req.name,
                "state": req.state,
                "district": req.district,
                "ward": req.ward,
                "preferred_language": req.language
            }).execute()
        return {"success": True, "message": "Citizen logged in successfully"}
    except Exception as e:
        print(f"[Citizen Auth Error]: {e}")
        return {"success": True, "message": "Fallback session continued"}

@router.get("/user-grievances")
def get_user_grievances(phone: str):
    try:
        res = supabase.table("grievances").select("*").eq("citizen_phone", phone).order("created_at", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"[Fetch Grievances Error]: {e}")
        return []

@router.post("/submit-grievance")
async def submit_grievance(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    user_name: str = Form("Nagrik"),
    user_phone: str = Form(""),
    user_ward: str = Form("Ward 1"),
    user_district: str = Form("Bhopal"),
    user_state: str = Form("Madhya Pradesh")
):
    try:
        print(f"[1] Receiving grievance from {user_name} ({user_phone}, {user_ward}, {user_district})...")
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

        if not triage.get("is_evidence_verified", True):
            reason = triage.get("verification_reason", "Photo aapki samasya se match nahi ho rahi hai.")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason}. Kripya sahi photo khinchein.",
                "transcript": transcript,
                "triage": triage
            }

        print("[4] Saving verified ticket to Supabase...")
        data = {
            "transcript": transcript,
            "department": triage.get("department", "Other"),
            "urgency": triage.get("urgency", 3),
            "summary": triage.get("summary", ""),
            "audio_url": audio_url,
            "image_url": image_url,
            "citizen_phone": user_phone,
            "citizen_ward": user_ward,
            "citizen_district": user_district,
            "citizen_state": user_state,
            "status": "Pending"
        }
        res = supabase.table("grievances").insert(data).execute()
        record_id = res.data[0]["id"] if res.data else "LOCAL-TX-OK"

        return {
            "success": True,
            "status": "FILED",
            "ticket_id": record_id,
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