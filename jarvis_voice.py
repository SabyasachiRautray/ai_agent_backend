"""
JARVIS Voice Module
--------------------
Adds Speech-to-Text (listening), Text-to-Speech (speaking), wake-word
activation, and Hindi/English bilingual support around your existing
LangChain conversational chain.

pip install SpeechRecognition edge-tts playsound==1.2.2 pyaudio
  -> Use Python 3.11 or 3.12 in this env -- pyaudio ships prebuilt wheels
     for those versions, so this installs cleanly with no compiler needed.
  -> Pin playsound to 1.2.2 specifically -- 1.3.x has a broken Windows backend.
"""

import os
import re
import time
import asyncio
import tempfile

import speech_recognition as sr
import edge_tts
from playsound import playsound
from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WAKE_WORD = "jarvis"
DEACTIVATE_PHRASES = {"deactivate", "go to sleep", "sleep now", "sleep"}
EXIT_PHRASES = {"exit", "quit", "shutdown", "shut down", "goodbye jarvis"}
HINDI_MODE_PHRASES = {"speak in hindi", "switch to hindi", "hindi mode", "hindi me baat karo"}
ENGLISH_MODE_PHRASES = {"speak in english", "switch to english", "english mode"}

VOICES = {
    "en": "en-GB-RyanNeural",
    "hi": "hi-IN-MadhurNeural",
}

# All command phrases combined -- checked in English regardless of the
# current language mode, since commands are always spoken in English.
COMMAND_PHRASES = DEACTIVATE_PHRASES | EXIT_PHRASES | HINDI_MODE_PHRASES | ENGLISH_MODE_PHRASES

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def matches_any(normalized_text: str, phrase_set: set[str]) -> bool:
    """True if any known phrase appears as a substring of what was said --
    e.g. "switch to hindi mode" still matches "switch to hindi". Using
    substring matching instead of exact equality means small variations
    in phrasing ("please deactivate", "hindi mode now") still get caught.
    """
    return any(phrase in normalized_text for phrase in phrase_set)


# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------
def _listen_raw(timeout: int, phrase_time_limit: int):
    """Records from the mic and returns the raw sr.AudioData, or None on
    silence/timeout."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            return recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return None


def listen_for_wake_word(timeout: int = 4, phrase_time_limit: int = 3) -> bool:
    """Short listen just to check for the wake word. Cheap and fast --
    only does English recognition since 'jarvis' is an English word."""
    print(f"💤 Standing by... (say '{WAKE_WORD}' to activate)")
    audio = _listen_raw(timeout, phrase_time_limit)
    if audio is None:
        return False
    try:
        text = sr.Recognizer().recognize_google(audio, language="en-IN")
        return WAKE_WORD in text.lower()
    except (sr.UnknownValueError, sr.RequestError):
        return False


def listen_query(forced_language: str | None, timeout: int = 6, phrase_time_limit: int = 15):
    """Records a query and returns (text, lang) where lang is 'en' or 'hi'.
    Returns ("", None) if nothing was understood.

    IMPORTANT: regardless of forced_language, we always try an English pass
    first and check it against COMMAND_PHRASES. Commands (deactivate, exit,
    switch to english/hindi) are always spoken in English, even while in
    Hindi mode -- if we skipped straight to hi-IN recognition in that mode,
    Google's Hindi recognizer would transliterate "deactivate" into
    unmatchable Devanagari gibberish instead of failing, which is exactly
    the bug this avoids.

    If it's not a command, we fetch the actual content: using forced_language
    if set (one recognizer call), or auto-detecting (English first, falling
    back to Hindi only if English recognition genuinely fails -- real Hindi
    speech usually fails en-IN outright, since the phonemes don't map).
    """
    print("🎙️  Listening...")
    audio = _listen_raw(timeout, phrase_time_limit)
    if audio is None:
        return "", None

    recognizer = sr.Recognizer()

    # Step 1: always check for a command in English first.
    try:
        en_text = recognizer.recognize_google(audio, language="en-IN")
    except (sr.UnknownValueError, sr.RequestError):
        en_text = None

    if en_text and matches_any(en_text.lower().strip(".,!? "), COMMAND_PHRASES):
        print(f"🗣️  (command) You said: {en_text}")
        return en_text, "en"

    # Step 2: not a command -> get the actual content in the right language.
    if forced_language == "hi":
        try:
            hi_text = recognizer.recognize_google(audio, language="hi-IN")
            print(f"🗣️  (Hindi) You said: {hi_text}")
            return hi_text, "hi"
        except (sr.UnknownValueError, sr.RequestError):
            return "", None

    if forced_language == "en":
        if en_text:
            print(f"🗣️  (English) You said: {en_text}")
            return en_text, "en"
        return "", None

    # Auto-detect (no forced language set).
    if en_text:
        print(f"🗣️  (English) You said: {en_text}")
        return en_text, "en"

    try:
        hi_text = recognizer.recognize_google(audio, language="hi-IN")
        print(f"🗣️  (Hindi) You said: {hi_text}")
        return hi_text, "hi"
    except (sr.UnknownValueError, sr.RequestError):
        return "", None


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------
async def _synthesize(text: str, voice: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def speak(text: str, lang: str = "en") -> None:
    """Synthesizes `text` with edge-tts (voice picked from VOICES by lang)
    and plays it via playsound. Never raises -- a TTS hiccup gets logged
    and skipped rather than crashing the whole assistant."""
    if not text.strip():
        return

    voice = VOICES.get(lang, VOICES["en"])

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        out_path = f.name

    try:
        asyncio.run(_synthesize(text, voice, out_path))
        playsound(out_path)
        time.sleep(0.4)  # let audio/room settle before we start listening again --
                          # otherwise the mic can catch the tail of our own playback
    except Exception as e:
        print(f"⚠️  TTS failed ({e}) -- continuing without speaking that line.")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


# ---------------------------------------------------------------------------
# Voice loop -- wake word activation + Hindi/English support
# ---------------------------------------------------------------------------
def get_reply_in_language(
    conversational_chain, query: str, reply_language: str, history: list | None = None, max_attempts: int = 2
) -> str:
    """Invokes the chain, and if the response's script doesn't match the
    requested language, retries once with a more forceful instruction.
    LLMs don't always obey a polite "respond in {language}" -- this catches
    the mismatch and pushes back rather than silently returning wrong-language
    text, which is what caused JARVIS to keep answering in Hindi after being
    told to switch to English.

    history: list of LangChain BaseMessage (HumanMessage/AIMessage) from
    recent turns, so follow-ups like "at 8pm" resolve against "book an
    appointment" from the prior turn. Optional -- defaults to no memory.
    """
    wants_hindi = reply_language.lower() == "hindi"
    text_response = ""
    history = history or []

    for attempt in range(max_attempts):
        lang_instruction = reply_language
        if attempt > 0:
            lang_instruction = (
                f"{reply_language} ONLY. This is mandatory -- do not use any other "
                f"language, not even a single word. Rewrite your answer entirely in {reply_language}."
            )
        response = conversational_chain.invoke({"query": query, "language": lang_instruction, "history": history})
        text_response = response.content if hasattr(response, "content") else str(response)

        has_devanagari = bool(DEVANAGARI_RE.search(text_response))
        if has_devanagari == wants_hindi:
            break  # response matches what we asked for

    return text_response


def voice_loop(conversational_chain, llm=None) -> None:
    """
    conversational_chain: a LangChain runnable (prompt | llm) that accepts
    {"query": ..., "language": ..., "history": [...]} and returns PLAIN TEXT
    (not structured Pydantic output). "language" should be "English" or
    "Hindi" -- the prompt uses it to decide which language to reply in.

    llm: the raw chat model, used for appointment-booking detail extraction
    ("book an appointment" -> collects name/email/doctor/date/time -> emails
    a confirmation). Pass None to disable booking (falls through to normal
    conversation for booking-sounding phrases instead).
    """
    from booking import (
        BOOKING_CANCEL_PHRASES, is_booking_trigger,
        extract_appointment_details, send_confirmation_email,
    )

    print(f"JARVIS loaded. Say '{WAKE_WORD}' to activate. Ctrl+C to fully quit anytime.")

    active = False
    forced_language: str | None = None  # None = auto-detect per query
    conversation_history: list = []  # HumanMessage/AIMessage pairs, most recent last
    MAX_HISTORY_MESSAGES = 12  # ~6 exchanges -- keeps context relevant without ballooning tokens
    pending_booking = False  # True while collecting appointment details across turns
    current_booking = None  # accumulates AppointmentDetails across turns -- never wiped by a failed parse

    while True:
        if not active:
            if listen_for_wake_word():
                active = True
                forced_language = None
                speak("Yes, sir. I'm listening.", lang="en")
            continue

        # --- active mode ---
        query, lang = listen_query(forced_language)
        if not query:
            continue

        normalized = query.lower().strip(".,!? ")

        if matches_any(normalized, DEACTIVATE_PHRASES):
            active = False
            speak(f"Going to sleep. Say {WAKE_WORD} to wake me.", lang="en")
            continue

        if matches_any(normalized, EXIT_PHRASES):
            speak("Shutting down completely. Goodbye, sir.", lang="en")
            break

        if matches_any(normalized, HINDI_MODE_PHRASES):
            forced_language = "hi"
            speak("Theek hai, ab main Hindi mein baat karunga.", lang="hi")
            continue

        if matches_any(normalized, ENGLISH_MODE_PHRASES):
            forced_language = "en"
            speak("Understood, sir. Switching to English.", lang="en")
            continue

        if llm is not None and pending_booking and matches_any(normalized, BOOKING_CANCEL_PHRASES):
            pending_booking = False
            current_booking = None
            speak("Booking cancelled, sir.", lang="en")
            continue

        if llm is not None and (pending_booking or is_booking_trigger(normalized)):
            pending_booking = True
            current_booking = extract_appointment_details(llm, query, conversation_history, existing=current_booking)
            missing = current_booking.missing_fields()

            if missing:
                reply_text = f"To confirm the booking, could you tell me {', and '.join(missing)}?"
            else:
                sent = send_confirmation_email(current_booking)
                reply_text = (
                    f"Booking confirmed, sir. I've sent a confirmation to {current_booking.patient_email}."
                    if sent else
                    "I have all the details, sir, but the confirmation email couldn't be sent -- "
                    "please check the email setup."
                )
                pending_booking = False
                current_booking = None

            print(f"🤖 JARVIS (booking): {reply_text}")
            speak(reply_text, lang="en")
            conversation_history.append(HumanMessage(content=query))
            conversation_history.append(AIMessage(content=reply_text))
            conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]
            continue

        reply_language = "Hindi" if lang == "hi" else "English"
        text_response = get_reply_in_language(conversational_chain, query, reply_language, history=conversation_history)

        # Pick the TTS voice from what the response ACTUALLY contains, not
        # just the detected input language -- if the model still slips into
        # the other language after retrying, this keeps voice matched to
        # the text instead of erroring out.
        actual_lang = "hi" if DEVANAGARI_RE.search(text_response) else "en"

        print(f"🤖 JARVIS ({reply_language}): {text_response}")
        speak(text_response, lang=actual_lang)

        conversation_history.append(HumanMessage(content=query))
        conversation_history.append(AIMessage(content=text_response))
        conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    # --- Example wiring against your existing setup ---
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-7B-Instruct:together",
        api_key=HF_TOKEN,
        base_url="https://router.huggingface.co/v1",
        temperature=0.3,
        max_tokens=512,
    )

    VOICE_SYSTEM_PROMPT = """You are JARVIS, a highly capable, witty, and unfailingly polite AI
assistant, in the style of Tony Stark's AI. Address the user as "Sir". Keep replies to 2-4
sentences since they will be spoken aloud -- no bullet points, no markdown, no JSON.
Respond entirely in {language}. If Hindi, use natural spoken Hindi in Devanagari script,
not overly formal or literary."""

    voice_prompt = ChatPromptTemplate.from_messages([
        ("system", VOICE_SYSTEM_PROMPT),
        ("human", "{query}"),
    ])

    conversational_chain = voice_prompt | llm

    voice_loop(conversational_chain)