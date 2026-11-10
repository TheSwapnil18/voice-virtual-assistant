@echo off
cd /d "%~dp0"
echo Starting Voice Virtual Assistant at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
