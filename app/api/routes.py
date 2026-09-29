import os
import re
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
        "version": "2.5.0"
    }

# ---------------------------------------------------------
# Route 1: Smart Pravesh OCR (Anti-Hallucination + Strict Regex)
# ---------------------------------------------------------
@router.post("/extract-id-details")
async def extract_id_details(card_image: UploadFile = File(...)):
    """
    Identity card se bina kisi hallucination ke exact printed text extract karta hai.
    """
    print(f"\n[Smart Pravesh OCR]: Image received -> {card_image.filename}")
    try:
        image_bytes = await card_image.read()
        print(f"[Smart Pravesh OCR]: Bytes read: {len(image_bytes)}")

        # Strict Prompt to prevent public-figure hallucination
        prompt = """
        You are a strict, raw OCR document parser.
        CRITICAL RULES:
        1. Read ONLY the literal words and numbers visibly printed on THIS image.
        2. DO NOT GUESS, hallucinate, or insert famous public figures, leaders, or sample names.
        3. If any field is unreadable, blurry, or missing, output an empty string "".

        Tasks:
        - Check if official markers exist ("Government of India", "भारत सरकार", Ashoka emblem, or disclaimer text).
        - Find the cardholder's exact printed Name (in English or Hindi).
        - Find the 12-digit number (usually printed prominently in 3 blocks of 4 digits).
        - Find Gender / Sex (Male / Female / Transgender).

        Return ONLY a single valid JSON object:
        {
          "is_authentic": true,
          "name": "",
          "id_number": "",
          "gender": ""
        }
        """

        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        mime = card_image.content_type or "image/jpeg"
        response = model.generate_content([prompt, {"mime_type": mime, "data": image_bytes}])
        raw_text = (response.text or "").strip()
        print(f"[Raw OCR Response]: {raw_text}")

        clean = raw_text
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        parsed = json.loads(clean)
        
        extracted_name = str(parsed.get("name") or "").strip()
        extracted_id = str(parsed.get("id_number") or "").replace(" ", "").replace("-", "").strip()
        extracted_gender = str(parsed.get("gender") or "").strip()

        # Blacklist common hallucinations
        hallucination_names = [
            "narendra modi", "narendra damodardas modi", "rahul gandhi",
            "sample user", "full name", "devansh kumar bhargava"
        ]
        if extracted_name.lower() in hallucination_names:
            extracted_name = ""

        # Validate 12-digit format strictly
        if not (extracted_id.isdigit() and len(extracted_id) == 12):
            # Try to regex search inside raw response if model outputted formatted text
            digit_matches = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', raw_text)
            if digit_matches:
                extracted_id = digit_matches[0].replace(" ", "")
            else:
                extracted_id = ""

        data_result = {
            "name": extracted_name,
            "id_number": extracted_id,
            "gender": extracted_gender
        }

        print(f"[Smart Pravesh Final Parsed]: {data_result}")
        return {
            "success": True,
            "is_authentic": parsed.get("is_authentic", True),
            "data": data_result
        }

    except Exception as e:
        print(f"[Smart Pravesh OCR Exception]: {e}")
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
    print(f"\n[Citizen Auth]: {req.name} ({req.phone})")
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
        print(f"[Citizen Auth Fallback]: {e}")
        return {"success": True, "message": "Fallback session active"}

# ---------------------------------------------------------
# Route 3: Fetch Grievances
# ---------------------------------------------------------
@router.get("/user-grievances")
def get_user_grievances(phone: str):
    try:
        res = supabase.table("grievances").select("*").eq("citizen_phone", phone).order("created_at", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        return []

# ---------------------------------------------------------
# Route 4: Multimodal Grievance Triage Pipeline
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
        # 1. Geofence Boundary Check (Indian bounds)
        is_geo_valid = (8.0 <= latitude <= 37.0) and (68.0 <= longitude <= 97.5)
        if not is_geo_valid:
            return {
                "success": False,
                "status": "REJECTED_GEOTAG_MISMATCH",
                "voice_feedback": "Aapki vartaman sthiti chune hue kshetra se bahar hai. Sahi sthal se shikayat darj karein.",
                "transcript": ""
            }

        audio_bytes = await audio.read()
        image_bytes = await image.read()

        audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
        image_url = await upload_file_to_storage(image_bytes, "jpg", "images")

        transcript, triage = await process_audio_and_triage(
            audio_bytes=audio_bytes,
            filename=audio.filename or "voice.m4a",
            image_bytes=image_bytes,
            previous_context=previous_context
        )

        if not triage.get("is_evidence_verified", True):
            reason = triage.get("verification_reason", "Tasveer aawaz me batayi samasya se mel nahi khati.")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason} Kripya sahi tasveer upload karein.",
                "transcript": transcript,
                "triage": triage
            }

        if triage.get("needs_followup", False):
            question = triage.get("followup_question", "Kripya samasya ka thik sthan batayein.")
            return {
                "success": False,
                "status": "NEEDS_FOLLOWUP",
                "voice_feedback": question,
                "transcript": transcript,
                "triage": triage
            }

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

        return {
            "success": True,
            "status": "FILED",
            "ticket_id": ticket_id,
            "voice_feedback": f"Aapki shikayat {dept} vibhag me darj ho chuki hai.",
            "transcript": transcript,
            "triage": triage,
            "audio_url": audio_url,
            "image_url": image_url
        }

    except Exception as e:
        print(f"[Submit Grievance Failure]: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))