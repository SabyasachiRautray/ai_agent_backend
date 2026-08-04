"""
FastAPI backend for the JARVIS React frontend.

Fully stateless -- no DB, no sessions, no login. The frontend holds
conversation history, pending_booking, and current_booking in its own
state (Redux) and sends them with every request; this backend just
processes one turn and returns the updated state for the frontend to keep.

pip install fastapi uvicorn python-multipart

Run: uvicorn api:app --reload --port 8000
"""

import os
import base64
import tempfile
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import speech_recognition as sr
import edge_tts
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from rag import HospitalRagChain
from jarvis_voice import get_reply_in_language, DEVANAGARI_RE, VOICES, matches_any
from booking import (
    AppointmentDetails, BOOKING_CANCEL_PHRASES, is_booking_trigger,
    extract_appointment_details, send_confirmation_email,
)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct:together",
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.3,
    max_tokens=512,
)
conversational_chain = HospitalRagChain(llm)

app = FastAPI(title="JARVIS API")

# Tighten allow_origins to your actual deployed frontend URL before shipping --
# "*" is fine for local dev, not for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response shapes
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ConverseResponse(BaseModel):
    transcript: str
    reply_text: str
    audio_base64: Optional[str]
    reply_lang: str  # "en" | "hi"
    pending_booking: bool
    current_booking: Optional[dict]
    history: list[ChatMessage]


# ---------------------------------------------------------------------------
# Helpers (mirrors gradio_app.py's logic, adapted for stateless HTTP)
# ---------------------------------------------------------------------------
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language="en-IN")
        except (sr.UnknownValueError, sr.RequestError):
            pass
        try:
            return recognizer.recognize_google(audio, language="hi-IN")
        except (sr.UnknownValueError, sr.RequestError):
            return ""
    finally:
        os.remove(tmp_path)


async def _synth(text: str, voice: str, out_path: str) -> None:
    await edge_tts.Communicate(text, voice).save(out_path)


def synthesize_to_base64(text: str, lang: str) -> Optional[str]:
    if not text.strip():
        return None
    voice = VOICES.get(lang, VOICES["en"])
    out_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        asyncio.run(_synth(text, voice, out_path))
        with open(out_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def to_lc_messages(history: list[ChatMessage]) -> list:
    messages = []
    for m in history:
        if m.role == "user":
            messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            messages.append(AIMessage(content=m.content))
    return messages[-12:]  # ~6 exchanges


# ---------------------------------------------------------------------------
# The one endpoint the frontend calls per turn
# ---------------------------------------------------------------------------
@app.post("/api/converse", response_model=ConverseResponse)
async def converse(
    audio: UploadFile = File(...),
    history_json: str = Form("[]"),          # JSON string: [{"role": "...", "content": "..."}]
    pending_booking: bool = Form(False),
    current_booking_json: Optional[str] = Form(None),  # JSON string of AppointmentDetails, or null
):
    import json

    history = [ChatMessage(**m) for m in json.loads(history_json)]
    current_booking = AppointmentDetails(**json.loads(current_booking_json)) if current_booking_json else None

    audio_bytes = await audio.read()
    query = transcribe_audio_bytes(audio_bytes)

    if not query:
        reply_text = "Sorry, sir -- I didn't catch that. Could you repeat?"
        return ConverseResponse(
            transcript="", reply_text=reply_text, audio_base64=synthesize_to_base64(reply_text, "en"),
            reply_lang="en", pending_booking=pending_booking, current_booking=current_booking,
            history=history,
        )

    lc_history = to_lc_messages(history)
    normalized = query.lower().strip(".,!? ")

    # --- booking cancel ---
    if pending_booking and matches_any(normalized, BOOKING_CANCEL_PHRASES):
        reply_text = "Booking cancelled, sir."
        new_history = history + [ChatMessage(role="user", content=query), ChatMessage(role="assistant", content=reply_text)]
        return ConverseResponse(
            transcript=query, reply_text=reply_text, audio_base64=synthesize_to_base64(reply_text, "en"),
            reply_lang="en", pending_booking=False, current_booking=None, history=new_history,
        )

    # --- booking flow ---
    if pending_booking or is_booking_trigger(normalized):
        details = extract_appointment_details(llm, query, lc_history, existing=current_booking)
        missing = details.missing_fields()

        if missing:
            reply_text = f"To confirm the booking, could you tell me {', and '.join(missing)}?"
            still_pending, next_booking = True, details
        else:
            sent = send_confirmation_email(details)
            reply_text = (
                f"Booking confirmed, sir. I've sent a confirmation to {details.patient_email}."
                if sent else
                "I have all the details, sir, but the confirmation email couldn't be sent -- please check the email setup."
            )
            still_pending, next_booking = False, None

        new_history = history + [ChatMessage(role="user", content=query), ChatMessage(role="assistant", content=reply_text)]
        return ConverseResponse(
            transcript=query, reply_text=reply_text, audio_base64=synthesize_to_base64(reply_text, "en"),
            reply_lang="en", pending_booking=still_pending,
            current_booking=next_booking.model_dump() if next_booking else None, history=new_history,
        )

    # --- normal RAG/chat ---
    reply_language = "Hindi" if DEVANAGARI_RE.search(query) else "English"
    answer = get_reply_in_language(conversational_chain, query, reply_language, history=lc_history)
    actual_lang = "hi" if DEVANAGARI_RE.search(answer) else "en"

    new_history = history + [ChatMessage(role="user", content=query), ChatMessage(role="assistant", content=answer)]
    return ConverseResponse(
        transcript=query, reply_text=answer, audio_base64=synthesize_to_base64(answer, actual_lang),
        reply_lang=actual_lang, pending_booking=pending_booking, current_booking=current_booking, history=new_history,
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}