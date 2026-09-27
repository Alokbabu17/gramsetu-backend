import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.services.ai_service import process_audio_and_triage
from app.services.db_service import upload_file_to_storage, supabase
from app.services.whatsapp_service import send_grievance_whatsapp_alert

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
        res = supabase.table("citizens").select("*").eq("phone", req.phone).execute()
        if res.data and len(res.data) > 0:
            supabase.table("citizens").update({
                "name": req.name,
                "state": req.state,
                "district": req.district,
                "ward": req.ward,
                "preferred_language": req.language
            }).eq("phone", req.phone).execute()
        else:
            supabase.table("citizens").insert({
                "phone": req.phone,
                "name": req.name,
                "state": req.state,
                "district": req.district,
                "ward": req.ward,
                "preferred_language": req.language
            }).execute()
        return {"success": True, "message": "Citizen logged in"}
    except Exception as e:
        print(f"[Citizen Auth Error]: {e}")
        return {"success": True, "message": "Fallback session continued"}

@router.get("/user-grievances")
def get_user_grievances(phone: str):
    try:
        res = supabase.table("grievances").select("*").eq("citizen_phone", phone).order("created_at", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"[Fetch Error]: {e}")
        return []

@router.post("/submit-grievance")
async def submit_grievance(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    user_name: str = Form("Nagrik"),
    user_phone: str = Form(""),
    user_ward: str = Form("Ward 1"),
    user_district: str = Form("Bhopal"),
    user_state: str = Form("Madhya Pradesh"),
    latitude: float = Form(23.2599),
    longitude: float = Form(77.4126),
    previous_context: str = Form("")
):
    try:
        print(f"[1] Receiving grievance: {user_name} ({user_phone}) from {user_district}, {user_state} | GPS: ({latitude}, {longitude})")
        audio_bytes = await audio.read()
        image_bytes = await image.read()

        # Geofence Validation
        is_geo_valid = (20.0 <= latitude <= 30.0) and (73.0 <= longitude <= 89.0)
        if not is_geo_valid:
            return {
                "success": False,
                "status": "REJECTED_GEOTAG_MISMATCH",
                "voice_feedback": "Aapki location darj kiye gaye rajya se bahar dikh rahi hai. Sahi sthan se shikayat darj karein.",
                "transcript": ""
            }

        print("[2] Uploading evidence files...")
        audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
        image_url = await upload_file_to_storage(image_bytes, "jpg", "images")

        print("[3] AI Multi-modal Vision & Speech Triage...")
        transcript, triage = await process_audio_and_triage(
            audio_bytes=audio_bytes,
            filename=audio.filename or "voice.m4a",
            image_bytes=image_bytes,
            previous_context=previous_context
        )

        if not triage.get("is_evidence_verified", True):
            reason = triage.get("verification_reason", "Photo aapki samasya se match nahi ho rahi hai.")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason} Kripya sahi photo khinchein.",
                "transcript": transcript,
                "triage": triage
            }

        if triage.get("needs_followup", False):
            question = triage.get("followup_question", "Kripya batayein ye samasya kaunse ward me hai?")
            return {
                "success": False,
                "status": "NEEDS_FOLLOWUP",
                "voice_feedback": question,
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
            "latitude": latitude,
            "longitude": longitude,
            "status": "Pending"
        }
        res = supabase.table("grievances").insert(data).execute()
        record_id = res.data[0]["id"] if res.data else "LOCAL-TX-OK"

        # WhatsApp alert trigger yahan chalta hai
        if user_phone:
            try:
                send_grievance_whatsapp_alert(
                    to_phone=user_phone,
                    citizen_name=user_name,
                    ticket_id=str(record_id),
                    department=triage.get('department', 'General'),
                    ward=user_ward,
                    district=user_district
                )
            except Exception as w_err:
                print(f"[WhatsApp Trigger Failed]: {w_err}")

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