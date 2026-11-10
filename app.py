"""
Main Web Server for the Voice-Based Virtual Assistant.

Connects all five case study stages into a simple REST API and serves the UI:
1. Audio Input (WAV analysis)
2. Speech Recognition (ASR)
3. Natural Language Processing (TF-IDF + Entity Extraction)
4. Task Execution & Speech Synthesis (Web/Windows Tasks + gTTS)
5. Word Error Rate (WER) Evaluation
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from modules.audio_input import analyze_wav_bytes
from modules.nlp_processor import IntentClassifier
from modules.speech_recognition import transcribe_wav_bytes
from modules.speech_synthesis import synthesize_speech
from modules.task_executor import WINDOWS_APPS, WEBSITE_URLS, execute_intent, is_local_windows_mode
from modules.wer_evaluator import calculate_wer

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Voice Virtual Assistant — Case Study", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

classifier = IntentClassifier(INTENTS_PATH)


class TextCommandRequest(BaseModel):
    text: str
    speak: bool = True


class WERRequest(BaseModel):
    reference: str
    hypothesis: str


@app.get("/")
def serve_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def get_status() -> dict:
    local_win = is_local_windows_mode()
    return {
        "mode": "local_windows" if local_win else "cloud_demo",
        "mode_label": (
            "Local Windows PC (Apps & Volume Live)"
            if local_win
            else "Cloud Demo (Web Links Live • Windows Tasks Simulated)"
        ),
        "supported_websites": sorted(WEBSITE_URLS.keys()),
        "supported_apps": sorted({info[0] for info in WINDOWS_APPS.values()}),
        "wer_benchmark_phrases": [
            "open youtube",
            "search youtube for python tutorial",
            "open google and search weather today",
            "open notepad",
            "open calculator",
            "increase the volume",
            "decrease the volume",
            "set volume to 50 percent",
            "mute the speakers",
            "what time is it",
        ],
    }


@app.post("/api/voice-command")
def process_voice_command(
    file: UploadFile = File(...),
    language: str = Form("en-US"),
    reference: str = Form(""),
) -> dict:
    wav_bytes = file.file.read()
    try:
        audio_info = analyze_wav_bytes(wav_bytes).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    asr_result = transcribe_wav_bytes(wav_bytes, language=language)
    if not asr_result.success:
        tts_err = synthesize_speech(asr_result.error_message)
        return {
            "audio": audio_info,
            "asr": asr_result.to_dict(),
            "nlp": None,
            "execution": {
                "action_type": "error",
                "status": "Speech not recognized",
                "reply_text": asr_result.error_message,
                "details": "Try speaking clearly closer to the microphone.",
                "target_url": "",
                "simulated": False,
                "volume_percent": None,
            },
            "tts": tts_err.to_dict(),
            "wer": None,
        }

    nlp_result = classifier.predict(asr_result.transcript)
    exec_result = execute_intent(nlp_result.intent, nlp_result.entities)
    tts_result = synthesize_speech(exec_result.reply_text)

    wer_result = None
    if reference.strip():
        wer_result = calculate_wer(reference, asr_result.transcript).to_dict()

    return {
        "audio": audio_info,
        "asr": asr_result.to_dict(),
        "nlp": nlp_result.to_dict(),
        "execution": exec_result.to_dict(),
        "tts": tts_result.to_dict(),
        "wer": wer_result,
    }


@app.post("/api/text-command")
def process_text_command(payload: TextCommandRequest) -> dict:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter a command.")

    nlp_result = classifier.predict(text)
    exec_result = execute_intent(nlp_result.intent, nlp_result.entities)
    tts_result = (
        synthesize_speech(exec_result.reply_text).to_dict()
        if payload.speak
        else {
            "success": False,
            "audio_base64": "",
            "mime_type": "audio/mpeg",
            "engine": "Skipped",
            "latency_ms": 0.0,
            "error_message": "",
        }
    )

    return {
        "audio": None,
        "asr": {
            "success": True,
            "transcript": text,
            "engine": "Typed Text Input (Bypassed ASR)",
            "latency_ms": 0.0,
            "error_message": "",
        },
        "nlp": nlp_result.to_dict(),
        "execution": exec_result.to_dict(),
        "tts": tts_result,
        "wer": None,
    }


@app.post("/api/wer")
def compute_wer_endpoint(payload: WERRequest) -> dict:
    try:
        return calculate_wer(payload.reference, payload.hypothesis).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
