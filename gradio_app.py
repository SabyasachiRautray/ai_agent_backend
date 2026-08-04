"""
Browser frontend for JARVIS -- voice in, voice out, no login/DB needed.

Records through the BROWSER's microphone (not pyaudio), which sidesteps
desktop mic/DLL issues entirely for the demo. Run this and open the local
URL Gradio prints; works on any machine with a browser mic permission.

pip install gradio
(all other deps are the ones you already installed for main.py / rag.py)
"""

import os
import tempfile
import asyncio

import speech_recognition as sr
import edge_tts
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag import HospitalRagChain
from jarvis_voice import get_reply_in_language, DEVANAGARI_RE, VOICES, matches_any
from langchain_core.messages import HumanMessage, AIMessage
from booking import BOOKING_CANCEL_PHRASES, is_booking_trigger, extract_appointment_details, send_confirmation_email

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


def transcribe_audio_file(filepath: str) -> str:
    """Transcribes a recorded audio file (from the browser mic) using the
    same auto-detect approach as jarvis_voice.py: English first, Hindi as
    fallback if English recognition fails outright."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio, language="en-IN")
    except (sr.UnknownValueError, sr.RequestError):
        pass

    try:
        return recognizer.recognize_google(audio, language="hi-IN")
    except (sr.UnknownValueError, sr.RequestError):
        return ""


async def _synthesize(text: str, voice: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize_reply(text: str, lang: str) -> str:
    voice = VOICES.get(lang, VOICES["en"])
    out_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    asyncio.run(_synthesize(text, voice, out_path))
    return out_path


MAX_HISTORY_MESSAGES = 12  # ~6 exchanges -- keeps context relevant without ballooning tokens


def chatbot_history_to_messages(history: list) -> list:
    """Converts the visible Gradio chatbot log (list of {"role", "content"}
    dicts) into LangChain messages for the chain's memory -- the chatbot's
    own display doubles as the conversation state, no separate store needed.
    Skips the "(unclear audio)" placeholder turns so they don't pollute context.
    """
    messages = []
    for turn in history:
        if turn.get("content") == "(unclear audio)":
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    return messages[-MAX_HISTORY_MESSAGES:]


def handle_turn(audio_filepath: str, history: list, pending_booking: bool, current_booking):
    if not audio_filepath:
        return history, None, None, pending_booking, current_booking

    query = transcribe_audio_file(audio_filepath)
    if not query:
        history = history + [
            {"role": "user", "content": "(unclear audio)"},
            {"role": "assistant", "content": "Sorry, sir -- I didn't catch that. Could you repeat?"},
        ]
        return history, None, None, pending_booking, current_booking

    lc_history = chatbot_history_to_messages(history)
    normalized = query.lower().strip(".,!? ")

    if pending_booking and matches_any(normalized, BOOKING_CANCEL_PHRASES):
        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": "Booking cancelled, sir."}]
        return history, synthesize_reply("Booking cancelled, sir.", "en"), None, False, None

    if pending_booking or is_booking_trigger(normalized):
        current_booking = extract_appointment_details(llm, query, lc_history, existing=current_booking)
        missing = current_booking.missing_fields()

        if missing:
            reply_text = f"To confirm the booking, could you tell me {', and '.join(missing)}?"
            still_pending = True
        else:
            sent = send_confirmation_email(current_booking)
            still_pending = False
            reply_text = (
                f"Booking confirmed, sir. I've sent a confirmation to {current_booking.patient_email}."
                if sent else
                "I have all the details, sir, but the confirmation email couldn't be sent -- please check the email setup."
            )
            current_booking = None

        audio_out = synthesize_reply(reply_text, "en")
        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": reply_text}]
        return history, audio_out, None, still_pending, current_booking

    reply_language = "Hindi" if DEVANAGARI_RE.search(query) else "English"
    answer = get_reply_in_language(conversational_chain, query, reply_language, history=lc_history)
    actual_lang = "hi" if DEVANAGARI_RE.search(answer) else "en"
    audio_out = synthesize_reply(answer, actual_lang)

    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    # Third return value clears mic_input so it's ready to record again --
    # without this, only the first question would ever get answered.
    return history, audio_out, None, pending_booking, current_booking


with gr.Blocks(title="JARVIS -- KLIMS Hospital Assistant") as demo:
    gr.Markdown("# JARVIS\n### KLIMS Hospital Voice Assistant -- ask about departments, doctors, symptoms, or anything else.")

    chatbot = gr.Chatbot(height=420)
    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Hold to record your question")
    audio_output = gr.Audio(label="JARVIS's reply", autoplay=True)
    pending_booking_state = gr.State(False)
    current_booking_state = gr.State(None)

    mic_input.stop_recording(
        handle_turn,
        inputs=[mic_input, chatbot, pending_booking_state, current_booking_state],
        outputs=[chatbot, audio_output, mic_input, pending_booking_state, current_booking_state],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),)