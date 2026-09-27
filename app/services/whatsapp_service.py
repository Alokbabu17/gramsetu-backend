import traceback
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
    """Citizen ko complaint darj hone par WhatsApp ya SMS alert bhejta hai."""
    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN):
        print("[Twilio]: Credentials not configured.")
        return False

    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

    clean_phone = to_phone.strip().replace(" ", "").replace("-", "")
    if not clean_phone.startswith("+"):
        if len(clean_phone) == 10:
            clean_phone = f"+91{clean_phone}"
        else:
            clean_phone = f"+{clean_phone}"

    sender_whatsapp = settings.TWILIO_WHATSAPP_NUMBER
    if not sender_whatsapp.startswith("whatsapp:"):
        sender_whatsapp = f"whatsapp:{sender_whatsapp}"

    target_whatsapp = f"whatsapp:{clean_phone}"

    ticket_short = ticket_id[:8]
    summary_msg = f"GramSetu: Namaste {citizen_name}, aapki shikayat #{ticket_short} ({department}) darj ho chuki hai. Ward: {ward}, {district}."

    # Step 1: Pre-approved sandbox template try karte hain (No Content API call)
    # Twilio Sandbox pre-approved template: "Your {{1}} code is {{2}}"
    try:
        msg = client.messages.create(
            from_=sender_whatsapp,
            to=target_whatsapp,
            body=f"Your GramSetu Ticket #{ticket_short} for {department} ({ward}, {district}) is registered successfully."
        )
        print(f"[Twilio WhatsApp Success]: SID {msg.sid} sent to {target_whatsapp}")
        return True

    except Exception as w_err:
        print(f"[Twilio WhatsApp Body Blocked by Meta/Trial]: {w_err}")

    # Step 2: Fallback to Direct Free SMS (Trial account phone par guaranteed deliver hota hai)
    try:
        raw_sender = settings.TWILIO_WHATSAPP_NUMBER.replace("whatsapp:", "")
        sms = client.messages.create(
            from_=raw_sender,
            to=clean_phone,
            body=summary_msg
        )
        print(f"[Twilio Direct SMS Fallback Success]: SID {sms.sid} sent to {clean_phone}")
        return True
    except Exception as s_err:
        print(f"[Twilio SMS Fallback Error]: {s_err}")

    return False