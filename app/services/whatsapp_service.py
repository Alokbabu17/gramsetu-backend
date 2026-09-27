from twilio.rest import Client
from app.core.config import settings

def send_grievance_whatsapp_alert(
    to_phone: str,
    citizen_name: str,
    ticket_id: str,
    department: str,
    ward: str,
    district: str
):
    """Citizen ko complaint darj hote hi WhatsApp par receipt bhejta hai."""
    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN):
        print("[Twilio]: Credentials not configured, skipping WhatsApp alert.")
        return False

    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        
        # Phone number formatting (+91 add karna agar na ho)
        clean_phone = to_phone.strip().replace(" ", "").replace("-", "")
        if not clean_phone.startswith("+"):
            if len(clean_phone) == 10:
                clean_phone = f"+91{clean_phone}"
            else:
                clean_phone = f"+{clean_phone}"

        message_body = (
            f"🏛️ *ग्रामसेतु AI (GramSetu) शिकायत रसीद*\n\n"
            f"नमस्ते *{citizen_name}* जी,\n"
            f"आपकी शिकायत सफलतापूर्वक दर्ज कर ली गई है।\n\n"
            f"📋 *टिकट संख्या:* #{ticket_id[:8]}\n"
            f"🏢 *संबंधित विभाग:* {department}\n"
            f"📍 *स्थान:* {ward}, {district}\n"
            f"⏳ *स्थिति:* लंबित (Pending - Under Review)\n\n"
            f"_हमारी टीम जल्द ही इस पर संज्ञान लेगी। अपनी शिकायत की स्थिति जांचने के लिए ग्रामसेतु ऐप खोलें।_"
        )

        message = client.messages.create(
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=f"whatsapp:{clean_phone}"
        )
        print(f"[Twilio WhatsApp Sent]: SID {message.sid} to {clean_phone}")
        return True
    except Exception as e:
        print(f"[Twilio WhatsApp Error]: {e}")
        return False