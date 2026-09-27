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
    """Citizen ko complaint receipt WhatsApp par bhejta hai."""
    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN):
        print("[Twilio]: Credentials not configured.")
        return False

    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

        clean_phone = to_phone.strip().replace(" ", "").replace("-", "")
        if not clean_phone.startswith("+"):
            if len(clean_phone) == 10:
                clean_phone = f"+91{clean_phone}"
            else:
                clean_phone = f"+{clean_phone}"

        sender = settings.TWILIO_WHATSAPP_NUMBER
        if not sender.startswith("whatsapp:"):
            sender = f"whatsapp:{sender}"

        recipient = f"whatsapp:{clean_phone}"

        msg_text = (
            f"🏛️ *ग्रामसेतु AI (GramSetu) शिकायत रसीद*\n\n"
            f"नमस्ते *{citizen_name}* जी,\n"
            f"आपकी शिकायत सफलतापूर्वक दर्ज कर ली गई है।\n\n"
            f"📋 *टिकट:* #{ticket_id[:8]}\n"
            f"🏢 *विभाग:* {department}\n"
            f"📍 *स्थान:* {ward}, {district}\n"
            f"⏳ *स्थिति:* लंबित (Pending)\n\n"
            f"_ग्रामसेतु टीम द्वारा जल्द ही संज्ञान लिया जाएगा।_"
        )

        try:
            # 1. Standard body message try karo
            msg = client.messages.create(
                from_=sender,
                to=recipient,
                body=msg_text
            )
            print(f"[Twilio WhatsApp Sent]: SID {msg.sid} to {recipient}")
            return True
        except Exception as inner_err:
            print(f"[Standard Send Failed, trying template fallback]: {inner_err}")
            # 2. Twilio default pre-approved appointment/alert template fallback
            # Agar account me ContentSid mandatory kar diya gaya hai
            content_templates = client.content.v1.contents.list(limit=5)
            if content_templates:
                sid = content_templates[0].sid
                msg = client.messages.create(
                    from_=sender,
                    to=recipient,
                    content_sid=sid,
                    content_variables='{"1":"' + citizen_name + '","2":"' + ticket_id[:8] + '"}'
                )
                print(f"[Twilio Template WhatsApp Sent]: SID {msg.sid}")
                return True
            raise inner_err

    except Exception as e:
        print(f"[Twilio WhatsApp Error]: {e}")
        traceback.print_exc()
        return False