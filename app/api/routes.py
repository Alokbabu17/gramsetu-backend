import os
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
# Health Check Route
# ---------------------------------------------------------
@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "gramsetu-core-backend",
        "version": "2.4.0"
    }

# ---------------------------------------------------------
# Route 1: Smart Pravesh OCR & Card Authenticity Extraction
# ---------------------------------------------------------
@router.post("/extract-id-details")
async def extract_id_details(card_image: UploadFile = File(...)):
    """
    Indian Government ID card se actual details extract karta hai bina kisi hardcoded dummy data ke.
    """
    print(f"\n[Smart Pravesh OCR]: Incoming image file -> {card_image.filename}")
    try:
        image_bytes = await card_image.read()
        print(f"[Smart Pravesh OCR]: Read {len(image_bytes)} bytes successfully.")

        candidate_models = ["gemini-3.1-flash-lite", "gemini-2.5-flash"]

        prompt = """
        You are an Indian Government ID OCR engine.
        Analyze this image carefully. Note that the card may be vertical, landscape, or upside down.

        Extract ONLY the real information printed on THIS specific document:
        1. Name: The actual full name of the cardholder printed on the card (written in English/Latin or Hindi script). Do NOT invent names.
        2. ID Number: The 12-digit Aadhaar / ID number printed in large bold digits, typically formatted as three groups of 4 digits (e.g. 4 digits space 4 digits space 4 digits). Strip all spaces and return exactly 12 digits. Do not pick barcode numbers, issue dates, or phone numbers.
        3. Gender: "Male", "Female", or "Transgender" based on what is printed next to Gender / Ling / Sex.
        4. is_authentic: Set to true if the card has government text like "Government of India", "भारत सरकार", or the standard disclaimer text.

        Return strict JSON only without any markdown formatting:
        {
          "is_authentic": true,
          "name": "",
          "id_number": "",
          "gender": ""
        }
        """

        raw_text = ""
        for model_name in candidate_models:
            try:
                print(f"[Smart Pravesh OCR]: Calling {model_name}...")
                model = genai.GenerativeModel(model_name)
                mime = card_image.content_type or "image/jpeg"
                response = model.generate_content([prompt, {"mime_type": mime, "data": image_bytes}])
                raw_text = (response.text or "").strip()
                if raw_text:
                    print(f"[Smart Pravesh OCR]: Response received from {model_name}.")
                    break
            except Exception as model_err:
                print(f"[Smart Pravesh OCR]: {model_name} failed: {model_err}")
                continue

        if not raw_text:
            return {"success": False, "is_authentic": False, "data": {"name": "", "id_number": "", "gender": ""}}

        clean = raw_text
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        parsed = json.loads(clean)
        print(f"[Smart Pravesh OCR Parsed]: {parsed}")

        name_val = str(parsed.get("name") or "").strip()
        id_val = str(parsed.get("id_number") or "").replace(" ", "").replace("-", "").strip()
        gender_val = str(parsed.get("gender") or "").strip()

        data_result = {
            "name": name_val,
            "id_number": id_val,
            "gender": gender_val
        }

        return {
            "success": True,
            "is_authentic": parsed.get("is_authentic", True),
            "data": data_result
        }

    except Exception as e:
        print(f"[Smart Pravesh OCR Exception]: {e}")
        traceback.print_exc()
        return {"success": False, "is_authentic": False, "data": {"name": "", "id_number": "", "gender": ""}}
# ---------------------------------------------------------
# Route 2: Citizen Login & Profile Upsert
# ---------------------------------------------------------
@router.post("/citizen-login")
def citizen_login(req: CitizenLoginRequest):
    print(f"\n[Citizen Auth]: Login initiated for {req.name} (Phone: {req.phone})")
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
            print(f"[Citizen Auth]: Updating existing record for {req.phone}")
            supabase.table("citizens").update(update_data).eq("phone", req.phone).execute()
        else:
            print(f"[Citizen Auth]: Creating fresh record for {req.phone}")
            update_data["phone"] = req.phone
            supabase.table("citizens").insert(update_data).execute()

        print("[Citizen Auth]: Session established successfully in Supabase.")
        return {"success": True, "message": "Citizen logged in successfully"}

    except Exception as e:
        print(f"[Citizen Auth Supabase Error - Safe Fallback]: {e}")
        traceback.print_exc()
        return {"success": True, "message": "Fallback session continued"}

# ---------------------------------------------------------
# Route 3: Fetch Citizen Grievance History
# ---------------------------------------------------------
@router.get("/user-grievances")
def get_user_grievances(phone: str):
    print(f"[Fetch Tickets]: Getting complaints list for citizen {phone}")
    try:
        res = supabase.table("grievances").select("*").eq("citizen_phone", phone).order("created_at", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"[Fetch Tickets Error]: {e}")
        traceback.print_exc()
        return []

# ---------------------------------------------------------
# Route 4: Complete Multimodal Grievance Triage Pipeline
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
    """
    Core AI Pipeline:
    1. Geofence Boundary Check (Latitude/Longitude validation)
    2. Upload raw audio & image evidence to Supabase Storage
    3. Multimodal cross-verification (Audio vs Image match)
    4. Follow-up inquiry generator for vague complaints
    5. Final ticket registration in Supabase Grievance table
    """
    try:
        print(f"\n=======================================================")
        print(f"[Grievance Intake]: Citizen {user_name} ({user_phone})")
        print(f"[Grievance Location]: {user_ward}, {user_district}, {user_state}")
        print(f"[Grievance GPS]: Lat: {latitude}, Long: {longitude}")
        print(f"=======================================================")

        # 1. Geofence Check for Indian Territory (8.0 to 37.0 N, 68.0 to 97.5 E)
        is_geo_valid = (8.0 <= latitude <= 37.0) and (68.0 <= longitude <= 97.5)
        if not is_geo_valid:
            print(f"[Geofence Violation]: Coordinates ({latitude}, {longitude}) outside valid boundary!")
            return {
                "success": False,
                "status": "REJECTED_GEOTAG_MISMATCH",
                "voice_feedback": "Aapki vartaman sthiti chune hue kshetra se mel nahi khati. Kripya sahi sthal se shikayat darj karein.",
                "transcript": ""
            }

        # 2. Read bytes for parallel processing
        audio_bytes = await audio.read()
        image_bytes = await image.read()

        print("[Storage]: Uploading evidence files to Supabase...")
        audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
        image_url = await upload_file_to_storage(image_bytes, "jpg", "images")
        print(f"[Storage Done]: Audio -> {audio_url} | Image -> {image_url}")

        # 3. AI Multimodal Triage (Audio Transcription + Vision Verification)
        print("[AI Engine]: Processing audio transcription and multimodal vision verification...")
        transcript, triage = await process_audio_and_triage(
            audio_bytes=audio_bytes,
            filename=audio.filename or "voice.m4a",
            image_bytes=image_bytes,
            previous_context=previous_context
        )
        print(f"[AI Transcript]: {transcript}")
        print(f"[AI Triage Output]: {triage}")

        # 4. Strict Evidence Cross-Validation Check
        if not triage.get("is_evidence_verified", True):
            reason = triage.get(
                "verification_reason",
                "Tasveer aapki aawaz me batayi gayi samasya se mel nahi kha rahi hai."
            )
            print(f"[Validation Failed]: {reason}")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason} Kripya samasya ki sachhi tasveer khinchein.",
                "transcript": transcript,
                "triage": triage
            }

        # 5. Missing Information / Follow-up Inquiry Check
        if triage.get("needs_followup", False):
            question = triage.get(
                "followup_question",
                "Kripya samasya ka thik sthan ya ward batayein taaki hum karwayi shuru kar sakein."
            )
            print(f"[Followup Required]: {question}")
            return {
                "success": False,
                "status": "NEEDS_FOLLOWUP",
                "voice_feedback": question,
                "transcript": transcript,
                "triage": triage
            }

        # 6. Save Verified Complaint Record in Supabase
        print("[Database]: Registering verified ticket in Supabase...")
        ticket_payload = {
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

        res = supabase.table("grievances").insert(ticket_payload).execute()
        ticket_id = res.data[0]["id"] if (res.data and len(res.data) > 0) else "TX-VERIFIED"
        dept = triage.get("department", "Sambandhit")

        print(f"[Database Success]: Ticket #{ticket_id} created for {dept} department.")

        return {
            "success": True,
            "status": "FILED",
            "ticket_id": ticket_id,
            "voice_feedback": f"Aapki shikayat safalta-purvak {dept} vibhag me darj ho chuki hai.",
            "transcript": transcript,
            "triage": triage,
            "audio_url": audio_url,
            "image_url": image_url
        }

    except Exception as e:
        print(f"[Submit Grievance Critical Failure]: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))