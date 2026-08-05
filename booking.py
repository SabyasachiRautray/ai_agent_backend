"""
Appointment booking: extracts booking details from the conversation using
the LLM, asks for anything missing, and emails a confirmation via Resend
(HTTPS email API) once complete.

Needs RESEND_API_KEY in .env -- get one free at https://resend.com.
Render's free tier blocks outbound SMTP (ports 25/465/587), so this uses
Resend's HTTPS API instead of smtplib -- works identically locally and
on Render without any platform upgrade.
"""

import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

RESEND_API_KEY = os.getenv("RESEND_API_KEY")

# This is a personal assistant for one fixed user -- no need to ask for
# name/email on every booking. Set these once in .env.
PATIENT_NAME = os.getenv("PATIENT_NAME")
PATIENT_EMAIL = os.getenv("PATIENT_EMAIL")

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[a-zA-Z]+")

# Exact-phrase matching missed natural variations like "book ME an
# appointment" or "book a pulmonologist appointment" (words in between break
# a substring match), silently falling through to normal chat instead of
# triggering the booking flow. Keyword-based detection is far more robust.
_BOOKING_ACTION_WORDS = ("book", "schedule", "confirm")
_BOOKING_TARGET_WORDS = ("appoint", "booking")  # "appoint" covers appointment/appointments


def is_booking_trigger(normalized_text: str) -> bool:
    has_action = any(word in normalized_text for word in _BOOKING_ACTION_WORDS)
    has_target = any(word in normalized_text for word in _BOOKING_TARGET_WORDS)
    return has_action and has_target


BOOKING_CANCEL_PHRASES = {"cancel booking", "cancel appointment", "never mind the booking", "never mind"}


class AppointmentDetails(BaseModel):
    patient_name: Optional[str] = None
    patient_email: Optional[str] = None
    doctor: Optional[str] = None
    department: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

    def missing_fields(self) -> list[str]:
        # Only doctor/department, date, and time vary per booking -- name
        # and email are fixed for this assistant's one user and filled in
        # automatically, so they're never asked for.
        missing = []
        if not self.doctor and not self.department:
            missing.append("which doctor or department")
        if not self.date:
            missing.append("the date")
        if not self.time:
            missing.append("the time")
        return missing


_parser = PydanticOutputParser(pydantic_object=AppointmentDetails)

_EXTRACTION_SYSTEM_PROMPT = """You extract appointment-booking details from a conversation between a
patient and JARVIS, a hospital voice assistant. Read the conversation history and the latest message,
and pull out whatever booking details have been mentioned so far across the WHOLE conversation --
doctor name, department, date, and time. This includes department or symptom-routing mentioned
EARLIER in the conversation, not just the most recent message (e.g. if Cardiology was already
discussed for chest pain, treat that as the department). Leave a field null if it genuinely
hasn't been mentioned anywhere. Do not guess or invent values.

CRITICAL: Respond with ONLY the raw JSON object below and absolutely nothing else -- no greeting,
no question to the patient, no explanation, no markdown code fences. You are not talking to the
patient here; you are only extracting data. Even if every field is null, still output valid JSON.

{format_instructions}
"""

_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", _EXTRACTION_SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
]).partial(format_instructions=_parser.get_format_instructions())


def extract_appointment_details(
    llm, query: str, history: list, existing: AppointmentDetails | None = None, max_attempts: int = 2
) -> AppointmentDetails:
    """Extracts booking details via the LLM and MERGES them into `existing`
    (details already collected in prior turns) -- never raises, and never
    wipes out what's already known. Small models occasionally ignore the
    JSON-only instruction and reply conversationally instead ("Sure, I can
    book that..."); when that happens we retry once with a sharper
    correction, and if it still fails, we simply learn nothing NEW this
    turn rather than discarding everything collected so far. Patient
    name/email are always filled in from .env -- this assistant has one
    fixed user, never extracted or asked for."""
    existing = existing or AppointmentDetails()
    chain = _extraction_prompt | llm

    new_details = None
    for attempt in range(max_attempts):
        try:
            raw = chain.invoke({"query": query, "history": history})
            raw_text = raw.content if hasattr(raw, "content") else str(raw)
            new_details = _parser.parse(raw_text)
            break
        except Exception as e:
            if attempt == 0:
                # Sharpen the instruction and retry once before giving up --
                # small models sometimes need a second, firmer nudge.
                query = (
                    f"{query}\n\n(Reminder: output ONLY the JSON object, no reply to the "
                    f"patient, no confirmation sentence -- just the extracted fields.)"
                )
            else:
                print(f"Booking extraction couldn't parse the model's output ({e}); keeping prior details.")

    if new_details is None:
        new_details = AppointmentDetails()  # nothing NEW learned this turn -- existing is preserved below

    merged = AppointmentDetails(
        patient_name=PATIENT_NAME,
        patient_email=PATIENT_EMAIL,
        doctor=new_details.doctor or existing.doctor,
        department=new_details.department or existing.department,
        date=new_details.date or existing.date,
        time=new_details.time or existing.time,
    )
    return merged


def send_confirmation_email(details: AppointmentDetails) -> bool:
    """Sends a booking confirmation email via Resend's HTTPS API. Returns
    True on success. Uses HTTPS instead of SMTP because Render's free tier
    blocks outbound traffic to SMTP ports (25/465/587) -- this works the
    same locally and on Render without needing a paid instance."""
    if not RESEND_API_KEY:
        print("RESEND_API_KEY missing in .env -- cannot send confirmation email.")
        return False

    body = (
        f"Dear {details.patient_name},\n\n"
        f"Your appointment has been confirmed at KLIMS:\n\n"
        f"Doctor / Department: {details.doctor or details.department}\n"
        f"Date: {details.date}\n"
        f"Time: {details.time}\n\n"
        f"Please arrive 20 minutes early with a valid photo ID.\n\n"
        f"-- KLIMS JARVIS Assistant"
    )

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={
                "from": "JARVIS <onboarding@resend.dev>",
                "to": [details.patient_email],
                "subject": "KLIMS Appointment Confirmation",
                "text": body,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send confirmation email: {e}")
        return False