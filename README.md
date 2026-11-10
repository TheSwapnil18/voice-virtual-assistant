# Voice-Based Virtual Assistant (Case Study Project)

A small, clean, college-level voice virtual assistant built with Python and FastAPI, featuring a minimal Gemini-style interface and built-in metrics for **Audio Input**, **Speech Recognition**, **Language Processing**, **Speech Synthesis**, and **Word Error Rate (WER)**.

---

## Project Structure

```text
.
├── app.py                   # FastAPI web server & REST API
├── data/
│   └── intents.json         # Training phrases for NLP intent matching
├── modules/
│   ├── __init__.py
│   ├── audio_input.py       # WAV parsing, loudness (dBFS), waveform peaks
│   ├── speech_recognition.py# Speech-to-text transcription (Google STT)
│   ├── nlp_processor.py     # TF-IDF cosine similarity + slot extraction
│   ├── speech_synthesis.py  # Text-to-speech MP3 generation (gTTS)
│   ├── task_executor.py     # Web URL builder & Windows app/volume control
│   └── wer_evaluator.py     # Levenshtein Word Error Rate (S, D, I, N)
├── static/
│   └── index.html           # Clean, minimal Gemini-style assistant UI + WER modal
├── tests/
│   ├── test_pipeline.py     # Automated unit & API tests
│   └── run_benchmark.py     # Reproducible clean vs noisy WER benchmark
├── CASE_STUDY_REPORT.md     # Complete academic case study report
├── start_assistant.bat      # One-click Windows launcher
├── .python-version          # Pins Python 3.11.11 for Render deployment
├── .gitignore               # Excludes .venv, caches, and generated files
└── requirements.txt         # Cross-platform dependencies
```

---

## What It Can Do

- **Open Websites:** *"open youtube"*, *"open google"*, *"open wikipedia"*, *"open github"*, or any `.com`/`.org`/`.in` domain.
- **Search YouTube & Google:** *"search youtube for lo-fi music"*, *"search google for word error rate"*.
- **Open Windows Apps (Local Windows Mode):** Notepad, Calculator, Paint, Command Prompt, File Explorer, Task Manager, Control Panel, Snipping Tool, Settings, Clock, Chrome, Edge.
- **Control Windows Volume (Local Windows Mode):** *"volume up"*, *"volume down"*, *"set volume to 50 percent"*, *"mute"*, *"unmute"*.
- **Case Study Inspection:** Click **`View pipeline details ▾`** under any assistant reply to inspect Audio, ASR, NLP, and TTS metrics, or click **`📊 WER Check`** in the top bar to evaluate Word Error Rate and export CSV logs.

---

## Run Locally on Windows

> **Note:** Python 3.11 is recommended (`SpeechRecognition` uses modules removed in Python 3.13+).

If you cloned this repository fresh:
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Then double-click **`start_assistant.bat`** (recommended on Windows so apps like Notepad open visibly on your desktop), or run:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000** in Chrome or Edge.

### Run Tests & WER Benchmark
```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe tests\run_benchmark.py
```

---

## Deploy a Free Global Link (Render — No Docker Needed)

1. Push this repository to GitHub (`.gitignore` excludes `.venv/` and caches automatically).
2. On [render.com](https://render.com), click **New + → Web Service** and select your GitHub repo.
3. Configure:
   - **Language:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** **Free**
4. Click **Deploy Web Service** to get your live HTTPS link (`https://<your-app>.onrender.com`).
   - Web tasks, voice ASR, NLP, TTS, and the WER Check work live on the cloud link; Windows-only desktop actions show **Cloud Demo** simulation mode.
