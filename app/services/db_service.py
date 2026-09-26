import uuid
from supabase import create_client, Client
from app.core.config import settings

# Trailing slash hata kar clean URL
clean_url = settings.SUPABASE_URL.rstrip("/")
supabase: Client = create_client(clean_url, settings.SUPABASE_KEY)

async def upload_file_to_storage(file_bytes: bytes, ext: str, folder: str) -> str:
    filename = f"{folder}/{uuid.uuid4()}.{ext}"
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "audio/m4a"
    
    supabase.storage.from_("complaints").upload(
        path=filename,
        file=file_bytes,
        file_options={"content-type": mime}
    )
    public_url = supabase.storage.from_("complaints").get_public_url(filename)
    return str(public_url)

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