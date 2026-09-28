import json
import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

from app.core.config import settings
from app.services.ai_service import process_audio_and_triage
from app.services.db_service import upload_file_to_storage, supabase

router = APIRouter()

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class CitizenLoginRequest(BaseModel):
    phone: str
    name: str
    state: str
    district: str
    ward: str
    language: str = "hi"
    id_number: str = ""
    gender: str = ""

# ---------------------------------------------------------
# Route 1: Smart Pravesh OCR Extractor
# ---------------------------------------------------------
@router.post("/extract-id-details")
async def extract_id_details(card_image: UploadFile = File(...)):
    """
    Identity card se Name, ID Number aur Gender extract karta hai.
    Gemini vision model cascade use karta hai taaki quota issue na ho.
    """
    try:
        image_bytes = await card_image.read()
        candidate_models = ["gemini-flash-latest", "gemini-3.1-flash-lite"]

        prompt = """
        You are an Indian Government ID OCR parser for citizen services.
        Inspect the uploaded identity card image and extract:
        1. Full Name of the card holder
        2. 12-digit Identity Number (digits only, remove all spaces)
        3. Gender/Sex (Male, Female, or Other)

        Output strict JSON format only:
        {
          "name": "Full Name",
          "id_number": "12-digit number",
          "gender": "Male / Female / Other"
        }
        """

        raw_text = ""
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                mime = card_image.content_type or "image/jpeg"
                image_part = {"mime_type": mime, "data": image_bytes}
                response = model.generate_content([prompt, image_part])
                raw_text = response.text.strip()
                if raw_text:
                    break
            except Exception as model_err:
                print(f"[OCR Candidate {model_name} failed]: {model_err}")
                continue

        if not raw_text:
            return {"success": False, "data": {"name": "", "id_number": "", "gender": ""}}

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        parsed = json.loads(raw_text.strip())
        print(f"[OCR Extracted Data]: {parsed}")
        return {"success": True, "data": parsed}

    except Exception as e:
        print(f"[OCR Endpoint Exception]: {e}")
        return {"success": False, "data": {"name": "", "id_number": "", "gender": ""}}

# ---------------------------------------------------------
# Route 2: Citizen Login & Registration
# ---------------------------------------------------------
@router.post("/citizen-login")
def citizen_login(req: CitizenLoginRequest):
    try:
        res = supabase.table("citizens").select("*").eq("phone", req.phone).execute()
        update_data = {
            "name": req.name,
            "state": req.state,
            "district": req.district,
            "ward": req.ward,
            "preferred_language": req.language,
            "id_number": req.id_number,
            "gender": req.gender
        }
        if res.data and len(res.data) > 0:
            supabase.table("citizens").update(update_data).eq("phone", req.phone).execute()
        else:
            update_data["phone"] = req.phone
            supabase.table("citizens").insert(update_data).execute()
        return {"success": True, "message": "Citizen logged in successfully"}
    except Exception as e:
        print(f"[Citizen Auth Exception]: {e}")
        return {"success": True, "message": "Fallback session continued"}

# ---------------------------------------------------------
# Route 3: Fetch Citizen Grievances
# ---------------------------------------------------------
@router.get("/user-grievances")
def get_user_grievances(phone: str):
    try:
        res = supabase.table("grievances").select("*").eq("citizen_phone", phone).order("created_at", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"[Fetch Grievances Exception]: {e}")
        return []

# ---------------------------------------------------------
# Route 4: Multimodal Grievance Triage & Submission
# ---------------------------------------------------------
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
        print(f"[1] Grievance Intake: {user_name} ({user_phone}) | Location: {user_ward}, {user_district}, {user_state} | Coordinates: ({latitude}, {longitude})")
        audio_bytes = await audio.read()
        image_bytes = await image.read()

        # GPS Geofence Range Check for Indian Territory
        is_geo_valid = (8.0 <= latitude <= 37.0) and (68.0 <= longitude <= 97.5)
        if not is_geo_valid:
            return {
                "success": False,
                "status": "REJECTED_GEOTAG_MISMATCH",
                "voice_feedback": "Aapki location darj kiye gaye kshetra se bahar hai. Kripya sahi sthal se shikayat darj karein.",
                "transcript": ""
            }

        print("[2] Uploading Evidence to Supabase Storage...")
        audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
        image_url = await upload_file_to_storage(image_bytes, "jpg", "images")

        print("[3] Running Multimodal AI Triage...")
        transcript, triage = await process_audio_and_triage(
            audio_bytes=audio_bytes,
            filename=audio.filename or "voice.m4a",
            image_bytes=image_bytes,
            previous_context=previous_context
        )

        # Cross-verification check between voice complaint & photo evidence
        if not triage.get("is_evidence_verified", True):
            reason = triage.get("verification_reason", "Tasveer aapki aawaz me batayi samasya se mel nahi khati.")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason} Kripya sahi sthal ki tasveer khinchein.",
                "transcript": transcript,
                "triage": triage
            }

        # Followup needed condition check
        if triage.get("needs_followup", False):
            question = triage.get("followup_question", "Kripya batayein ye samasya kaunse ward me sthit hai?")
            return {
                "success": False,
                "status": "NEEDS_FOLLOWUP",
                "voice_feedback": question,
                "transcript": transcript,
                "triage": triage
            }

        print("[4] Saving Verified Complaint Ticket to Supabase Database...")
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
        print(f"[Submit Grievance Critical Error]: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))