from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import json
import google.generativeai as genai
from app.core.config import settings
from app.services.ai_service import process_audio_and_triage
from app.services.db_service import upload_file_to_storage, supabase

router = APIRouter()

class CitizenLoginRequest(BaseModel):
    phone: str
    name: str
    state: str
    district: str
    ward: str
    language: str = "hi"
    id_number: str = ""
    gender: str = ""

@router.post("/extract-id-details")
async def extract_id_details(card_image: UploadFile = File(...)):
    """Identity card image se details extract karta hai."""
    try:
        image_bytes = await card_image.read()
        model = genai.GenerativeModel("gemini-3.1-flash-lite")

        prompt = """
        You are an Indian Government ID OCR parser.
        Inspect the uploaded document image and extract:
        1. Full Name of the person
        2. 12-digit Identity Number (clean digits without spaces)
        3. Gender/Sex (Male, Female, or Other)

        Output strict JSON only:
        {
          "name": "Extracted Name or empty",
          "id_number": "Extracted 12-digit ID or empty",
          "gender": "Male / Female / Other or empty"
        }
        """

        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        response = model.generate_content([prompt, image_part])
        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        parsed = json.loads(raw_text.strip())
        return {"success": True, "data": parsed}
    except Exception as e:
        print(f"[OCR Extraction Error]: {e}")
        return {"success": False, "data": {"name": "", "id_number": "", "gender": ""}}

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
        return {"success": True, "message": "Citizen logged in"}
    except Exception as e:
        print(f"[Citizen Auth Error]: {e}")
        return {"success": True, "message": "Fallback session continued"}