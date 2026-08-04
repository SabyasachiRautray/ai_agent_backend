"""
Appointment booking: extracts booking details from the conversation using
the LLM, asks for anything missing, and emails a confirmation via Gmail
SMTP once complete.

Needs GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env -- an App Password
(not your regular Gmail password), generated at
https://myaccount.google.com/apppasswords (requires 2-Step Verification).
"""

import os
import re
import smtplib
from email.mime.text import MIMEText
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

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
    """Sends a booking confirmation email via Gmail SMTP. Returns True on success."""
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("Gmail credentials missing in .env -- cannot send confirmation email.")
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
    msg = MIMEText(body)
    msg["Subject"] = "KLIMS Appointment Confirmation"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = details.patient_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send confirmation email: {e}")
        return False