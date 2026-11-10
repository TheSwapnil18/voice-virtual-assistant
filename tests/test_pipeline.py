"""Unit and API tests for all five stages of the Voice Virtual Assistant."""

from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from modules.audio_input import analyze_wav_bytes
from modules.nlp_processor import IntentClassifier
from modules.task_executor import execute_intent
from modules.wer_evaluator import calculate_wer

INTENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "intents.json"


def _make_sine_wav(duration_sec: float = 1.0, sample_rate: int = 16000, amp: float = 0.25) -> bytes:
    """Create a 16-bit mono PCM WAV tone in memory for testing."""
    num_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            sample = int(amp * 32767 * math.sin(2 * math.pi * 440.0 * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def test_audio_input_analysis() -> None:
    wav_bytes = _make_sine_wav(duration_sec=1.0, sample_rate=16000, amp=0.25)
    info = analyze_wav_bytes(wav_bytes)
    assert info.sample_rate_hz == 16000
    assert info.channels == 1
    assert info.sample_width_bits == 16
    assert info.duration_sec == 1.0
    assert info.quality_label == "Good"
    assert len(info.waveform_peaks) == 36


@pytest.mark.parametrize(
    ("utterance", "expected_intent", "expected_entities"),
    [
        ("open youtube", "open_website", {"site": "youtube"}),
        ("open you tube", "open_website", {"site": "youtube"}),
        ("please open google maps", "open_website", {"site": "google maps"}),
        ("search youtube for python tutorial", "search_youtube", {"query": "python tutorial"}),
        ("search google for word error rate", "search_google", {"query": "word error rate"}),
        ("open notepad", "open_app", {"app": "notepad"}),
        ("open note pad", "open_app", {"app": "notepad"}),
        ("launch calculator", "open_app", {"app": "calculator"}),
        ("turn up the volume", "volume_up", {}),
        ("decrease the volume", "volume_down", {}),
        ("set volume to 45 percent", "volume_set", {"volume_percent": 45}),
        ("mute the speakers", "volume_mute", {}),
        ("unmute volume", "volume_unmute", {}),
        ("what time is it", "get_time", {}),
        ("asdfghjkl qwerty zzz", "unknown", {}),
    ],
)
def test_nlp_intent_and_slots(utterance: str, expected_intent: str, expected_entities: dict) -> None:
    classifier = IntentClassifier(INTENTS_PATH)
    res = classifier.predict(utterance)
    assert res.intent == expected_intent
    for key, val in expected_entities.items():
        assert res.entities.get(key) == val


def test_wer_calculations() -> None:
    exact = calculate_wer("open youtube", "Open YouTube!")
    assert exact.wer_percent == 0.0
    assert exact.substitutions == 0
    assert exact.deletions == 0
    assert exact.insertions == 0

    pct_norm = calculate_wer("set volume to 50 percent", "set volume to 50%")
    assert pct_norm.wer_percent == 0.0

    sub_case = calculate_wer("open youtube", "open google")
    assert sub_case.substitutions == 1
    assert sub_case.total_ref_words == 2
    assert sub_case.wer_percent == 50.0

    del_case = calculate_wer("search youtube for python", "search for python")
    assert del_case.deletions == 1
    assert del_case.total_ref_words == 4
    assert del_case.wer_percent == 25.0

    ins_case = calculate_wer("open notepad", "please open notepad now")
    assert ins_case.insertions == 2
    assert ins_case.total_ref_words == 2
    assert ins_case.wer_percent == 100.0


def test_task_executor_simulated_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULATE_WINDOWS", "true")

    web_res = execute_intent("open_website", {"site": "youtube"})
    assert web_res.target_url == "https://www.youtube.com"
    assert web_res.simulated is False

    yt_res = execute_intent("search_youtube", {"query": "nlp tutorial"})
    assert "youtube.com/results" in yt_res.target_url
    assert "nlp+tutorial" in yt_res.target_url

    app_res = execute_intent("open_app", {"app": "notepad"})
    assert app_res.simulated is True
    assert "Notepad" in app_res.reply_text

    vol_res = execute_intent("volume_set", {"volume_percent": 60})
    assert vol_res.simulated is True
    assert vol_res.volume_percent == 60


def test_fastapi_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULATE_WINDOWS", "true")
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert "Voice Assistant" in home.text

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["mode"] == "cloud_demo"

    text_cmd = client.post("/api/text-command", json={"text": "open youtube", "speak": False})
    assert text_cmd.status_code == 200
    body = text_cmd.json()
    assert body["nlp"]["intent"] == "open_website"
    assert body["execution"]["target_url"] == "https://www.youtube.com"

    wer_res = client.post("/api/wer", json={"reference": "open youtube", "hypothesis": "open your tube"})
    assert wer_res.status_code == 200
    assert wer_res.json()["wer_percent"] == 100.0
