# VibeKnowing Worker

A FastAPI worker service for video transcription using yt-dlp and OpenAI Whisper.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key in `worker.py`

3. Run the worker:
```bash
uvicorn worker:app --host 0.0.0.0 --port 8000
```

## Usage

The worker exposes a `/transcribe` endpoint that accepts video URLs and returns transcripts.

## Requirements

- Python 3.8+
- ffmpeg (for audio processing)
- yt-dlp (for video downloading)
- OpenAI API key