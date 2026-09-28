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
# Route 1: Smart Pravesh OCR & Authenticity Verifier
# ---------------------------------------------------------
@router.post("/extract-id-details")
async def extract_id_details(card_image: UploadFile = File(...)):
    """
    Identity card se Name, ID Number aur Gender extract karta hai
    aur mandatory disclaimer text check karke authenticity verify karta hai.
    """
    try:
        image_bytes = await card_image.read()
        candidate_models = ["gemini-flash-latest", "gemini-3.1-flash-lite"]

        prompt = """
        You are an Indian Government ID OCR and Authenticity Verification tool.
        Examine this document photo carefully (Note: the image might be rotated or vertical).

        Task 1: Verify Authenticity
        Check if the document contains standard government authenticity text, specifically:
        - "Aadhaar is proof of identity, not of citizenship" or "आधार पहचान का प्रमाण है"
        - Or "Government of India" / "भारत सरकार"
        Set "is_authentic" to true if these official markers/disclaimers exist, else false.

        Task 2: Extract Details
        1. Full Name of the individual (person's actual name, e.g. "Devansh Kumar Bhargava", not government titles).
        2. 12-digit numeric identity number (extract clean 12 digits, remove spaces).
        3. Gender / Sex (Male / Female / Transgender).

        Respond with STRICT JSON only in this exact format:
        {
          "is_authentic": true,
          "name": "Full Name",
          "id_number": "12-digit number",
          "gender": "Male"
        }

        Do not output markdown explanations. If blurry, put empty string "" for fields.
        """

        raw_text = ""
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                mime = card_image.content_type or "image/jpeg"
                image_part = {"mime_type": mime, "data": image_bytes}
                response = model.generate_content([prompt, image_part])
                raw_text = (response.text or "").strip()
                if raw_text:
                    break
            except Exception as model_err:
                print(f"[OCR Candidate {model_name} failed]: {model_err}")
                continue

        if not raw_text:
            return {
                "success": False,
                "is_authentic": False,
                "message": "Model response empty",
                "data": {"name": "", "id_number": "", "gender": ""}
            }

        clean_json = raw_text
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        parsed = json.loads(clean_json)
        print(f"[OCR Verification Result]: {parsed}")

        is_authentic = parsed.get("is_authentic", True)
        extracted_name = str(parsed.get("name") or "").strip()
        extracted_id = str(parsed.get("id_number") or "").replace(" ", "").replace("-", "").strip()
        extracted_gender = str(parsed.get("gender") or "").strip()

        data_result = {
            "name": "" if extracted_name.lower() in ["none", "null", "full name"] else extracted_name,
            "id_number": "" if extracted_id.lower() in ["none", "null"] else extracted_id,
            "gender": "" if extracted_gender.lower() in ["none", "null"] else extracted_gender
        }

        return {
            "success": True,
            "is_authentic": is_authentic,
            "data": data_result
        }

    except Exception as e:
        print(f"[OCR Critical Error]: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "is_authentic": False,
            "data": {"name": "", "id_number": "", "gender": ""}
        }

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
        print(f"[1] Grievance Intake: {user_name} ({user_phone}) | Location: {user_ward}, {user_district}, {user_state} | GPS: ({latitude}, {longitude})")
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

        if not triage.get("is_evidence_verified", True):
            reason = triage.get("verification_reason", "Tasveer aapki aawaz me batayi samasya se mel nahi khati.")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason} Kripya sahi sthal ki tasveer khinchein.",
                "transcript": transcript,
                "triage": triage
            }

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