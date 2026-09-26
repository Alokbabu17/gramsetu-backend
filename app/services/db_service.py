import uuid
import re
from supabase import create_client, Client
from app.core.config import settings

# URL ko clean base format mein convert karo
raw_url = settings.SUPABASE_URL.strip().rstrip("/")
clean_url = re.sub(r"/rest/v1/?$", "", raw_url)

supabase: Client = create_client(clean_url, settings.SUPABASE_KEY)

async def upload_file_to_storage(file_bytes: bytes, ext: str, folder: str) -> str:
    try:
        filename = f"{uuid.uuid4()}.{ext}"
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "audio/m4a"
        
        supabase.storage.from_("complaints").upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": mime}
        )
        return str(supabase.storage.from_("complaints").get_public_url(filename))
    except Exception as e:
        print(f"[Storage Skip]: {e}")
        return f"https://placeholder-storage.org/{folder}_{uuid.uuid4()}.{ext}"

async def save_grievance(transcript: str, department: str, urgency: int, summary: str, audio_url: str, image_url: str):
    data = {
        "transcript": transcript,
        "department": department,
        "urgency": urgency,
        "summary": summary,
        "audio_url": audio_url,
        "image_url": image_url,
        "status": "Pending"
    }
    response = supabase.table("grievances").insert(data).execute()
    return response.data[0]