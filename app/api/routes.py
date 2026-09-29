import os
import re
import json
import math
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

class DarshiniChatRequest(BaseModel):
    message: str
    phone: str
    language: str = "hi"

# State Bounding Geofences for Strict Cross-Validation (Phase 3)
STATE_GEO_BOUNDS = {
    "Madhya Pradesh": {"lat_min": 21.0, "lat_max": 26.9, "lon_min": 74.0, "lon_max": 82.8},
    "Bihar": {"lat_min": 24.3, "lat_max": 27.6, "lon_min": 83.3, "lon_max": 88.3},
    "Uttar Pradesh": {"lat_min": 23.8, "lat_max": 30.5, "lon_min": 77.0, "lon_max": 84.7},
    "West Bengal": {"lat_min": 21.5, "lat_max": 27.3, "lon_min": 85.8, "lon_max": 89.9},
}

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Haversine formula to compute distance between two GPS coordinates in KM."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "gramsetu-core-backend", "version": "3.0.0"}

# ---------------------------------------------------------
# Route 1: Smart Pravesh OCR (Phase 1)
# ---------------------------------------------------------
@router.post("/extract-id-details")
async def extract_id_details(card_image: UploadFile = File(...)):
    try:
        image_bytes = await card_image.read()
        prompt = """
        You are an Indian Government ID OCR engine.
        Analyze this image carefully. Note that the card may be vertical, landscape, or upside down.

        Extract ONLY the real information printed on THIS specific document:
        1. Name: The actual full name of the cardholder printed on the card. Do NOT invent names.
        2. ID Number: The 12-digit Aadhaar / ID number printed in large bold digits. Strip spaces.
        3. Gender: "Male", "Female", or "Transgender" based on what is printed.
        4. is_authentic: Set to true if the card has government text like "Government of India", "भारत सरकार", or the standard disclaimer text.

        Return strict JSON only:
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

        # Prevent generic hallucinations
        if extracted_name.lower() in ["narendra modi", "sample user", "devansh kumar bhargava", "full name"]:
            extracted_name = ""

        if not (extracted_id.isdigit() and len(extracted_id) == 12):
            matches = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', raw_text)
            extracted_id = matches[0].replace(" ", "") if matches else ""

        return {
            "success": True,
            "is_authentic": parsed.get("is_authentic", True),
            "data": {
                "name": extracted_name,
                "id_number": extracted_id,
                "gender": extracted_gender
            }
        }
    except Exception as e:
        print(f"[OCR Error]: {e}")
        return {"success": False, "is_authentic": False, "data": {"name": "", "id_number": "", "gender": ""}}

# ---------------------------------------------------------
# Route 2: Citizen Login
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
        print(f"[Citizen Login Fallback]: {e}")
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
# Route 4: Darshini Chat & Civic Advisory (Phase 2)
# ---------------------------------------------------------
@router.post("/darshini-chat")
async def darshini_chat(req: DarshiniChatRequest):
    user_query = req.message.lower().strip()
    status_keywords = ["status", "complaint", "shikayat", "pending", "ticket", "haal", "stithi", "kya hua"]

    if any(k in user_query for k in status_keywords) and req.phone:
        try:
            res = supabase.table("grievances").select("*").eq("citizen_phone", req.phone).order("created_at", desc=True).limit(5).execute()
            records = res.data if res.data else []
            if not records:
                return {"reply": "Aapki koi bhi pending shikayat darj nahi mili hai. Aap nayi shikayat darj kar sakte hain.", "language": req.language}

            summary_items = []
            for r in records:
                t_id = str(r.get("id"))[:6]
                dept = r.get("department", "Prashasan")
                st = r.get("status", "Pending")
                is_m = " [MASTER TICKET]" if r.get("is_master") else ""
                summary_items.append(f"Ticket #{t_id}{is_m} ({dept}): {st}")

            reply = "Aapki shikayatein: " + ", ".join(summary_items) + ". Sambandhit vibhag ispar karwayi kar raha hai."
            return {"reply": reply, "language": req.language}
        except Exception as e:
            print(f"[Darshini Status Error]: {e}")

    try:
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        advisory_prompt = f"""
        You are 'Darshini', an empathetic, wise AI rural civic guide for GramSetu platform.
        Language target: {req.language} (hi for Hindi, en for English, ur for Urdu, bn for Bangla).

        User Query: "{req.message}"

        Your Mission:
        - Guide them with exact right steps for court cases, land disputes, RTI filing, police disputes, or civic issues.
        - Keep the reply concise (under 3-4 sentences), polite, actionable.
        - Respond in the language requested: {req.language}.
        """
        response = model.generate_content(advisory_prompt)
        return {"reply": (response.text or "").strip(), "language": req.language}
    except Exception as e:
        return {
            "reply": "Main aapki sahayata ke liye tayar hoon. Zameen vivad ke liye SDM karyalaya ya RTI ke liye sambandhit Jan Suchna Adhikari (PIO) se sampark karein.",
            "language": req.language
        }

# ---------------------------------------------------------
# Route 5: Multimodal Intake with Phase 3 (Geotag) & Phase 4 (Clustering)
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
        print(f"\n[Grievance Intake]: {user_name} ({user_phone}) from {user_ward}, {user_district}, {user_state}")
        print(f"[GPS Coordinates]: Lat: {latitude}, Long: {longitude}")

        # PHASE 3: Strict State Geofence Cross-Validation
        if user_state in STATE_GEO_BOUNDS:
            bounds = STATE_GEO_BOUNDS[user_state]
            if not (bounds["lat_min"] <= latitude <= bounds["lat_max"] and bounds["lon_min"] <= longitude <= bounds["lon_max"]):
                print(f"[Phase 3 Rejection]: Coordinates ({latitude}, {longitude}) do not match state {user_state}")
                return {
                    "success": False,
                    "status": "REJECTED_GEOTAG_MISMATCH",
                    "voice_feedback": f"Aapke chune hue rajya ({user_state}) aur vartaman GPS location me antar paya gaya hai. Sahi sthan se shikayat karein.",
                    "transcript": ""
                }

        audio_bytes = await audio.read()
        image_bytes = await image.read()

        audio_url = await upload_file_to_storage(audio_bytes, "m4a", "audios")
        image_url = await upload_file_to_storage(image_bytes, "jpg", "images")

        # Multimodal AI Triage
        transcript, triage = await process_audio_and_triage(
            audio_bytes=audio_bytes,
            filename=audio.filename or "voice.m4a",
            image_bytes=image_bytes,
            previous_context=previous_context
        )

        # PHASE 3: Cross-validation between Audio vs Image evidence
        if not triage.get("is_evidence_verified", True):
            reason = triage.get("verification_reason", "Tasveer aawaz me batayi samasya se mel nahi khati.")
            return {
                "success": False,
                "status": "REJECTED_EVIDENCE_MISMATCH",
                "voice_feedback": f"Dhyan dein, {reason} Kripya sahi tasveer upload karein.",
                "transcript": transcript,
                "triage": triage
            }

        # Follow-up check
        if triage.get("needs_followup", False):
            question = triage.get("followup_question", "Kripya samasya ka thik sthan batayein.")
            return {
                "success": False,
                "status": "NEEDS_FOLLOWUP",
                "voice_feedback": question,
                "transcript": transcript,
                "triage": triage
            }

        target_dept = triage.get("department", "Other")

        # PHASE 4: Duplicate Detection & Master Ticket Clustering
        # Check active complaints in the same department and district
        active_tickets = supabase.table("grievances")\
            .select("id, latitude, longitude, cluster_count, is_master, priority")\
            .eq("department", target_dept)\
            .eq("citizen_district", user_district)\
            .eq("status", "Pending")\
            .execute()

        cluster_master_id = None
        nearby_matches = []

        if active_tickets.data:
            for t in active_tickets.data:
                t_lat = t.get("latitude")
                t_lon = t.get("longitude")
                if t_lat and t_lon:
                    dist = calculate_distance_km(latitude, longitude, float(t_lat), float(t_lon))
                    # Agar distance 0.5 KM (500 meters) ke andar hai
                    if dist <= 0.5:
                        nearby_matches.append(t)

        is_master_ticket = False
        ticket_priority = "NORMAL"

        # Clustering condition: 5 ya usse zyada complaints same zone me hone par Master Ticket
        if len(nearby_matches) >= 4:  # Existing 4 + current 1 = 5
            is_master_ticket = True
            ticket_priority = "SUPER_HIGH_MASTER"
            # Purane ticket me se kisi ko master chuna ya naye ko Master banaya
            cluster_master_id = nearby_matches[0]["id"]
            # Purane tickets ka cluster_count aur priority update karo
            new_count = len(nearby_matches) + 1
            supabase.table("grievances").update({
                "cluster_count": new_count,
                "priority": "SUPER_HIGH_MASTER",
                "is_master": True
            }).eq("id", cluster_master_id).execute()
            print(f"[Phase 4 Clustering]: MASTER TICKET formed for {target_dept} with {new_count} grievances!")
        elif len(nearby_matches) > 0:
            cluster_master_id = nearby_matches[0]["id"]
            ticket_priority = "HIGH"

        # Insert new grievance
        ticket_payload = {
            "transcript": transcript,
            "department": target_dept,
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
            "status": "Pending",
            "is_master": is_master_ticket,
            "master_ticket_id": cluster_master_id,
            "cluster_count": len(nearby_matches) + 1,
            "priority": ticket_priority
        }

        res = supabase.table("grievances").insert(ticket_payload).execute()
        ticket_id = res.data[0]["id"] if (res.data and len(res.data) > 0) else "TX-OK"

        master_alert = " Is sthan se kai shikayatein aayi hain, ise Master Ticket banakar uchh prathmikta di gayi hai." if is_master_ticket else ""
        feedback_text = f"Aapki shikayat {target_dept} vibhag me darj ho chuki hai.{master_alert}"

        return {
            "success": True,
            "status": "FILED",
            "ticket_id": ticket_id,
            "is_master": is_master_ticket,
            "priority": ticket_priority,
            "voice_feedback": feedback_text,
            "transcript": transcript,
            "triage": triage,
            "audio_url": audio_url,
            "image_url": image_url
        }

    except Exception as e:
        print(f"[Grievance Intake Error]: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))