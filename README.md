# VibeKnowing Worker

A FastAPI worker service for video transcription using yt-dlp and OpenAI Whisper. This worker must run on a local machine (with a residential IP) to bypass YouTube restrictions and is exposed to the cloud backend via ngrok.

---

## Why Run the Worker Locally?
- **YouTube blocks data center/cloud IPs** for transcript and video downloads. Running the worker on your home/office network (residential IP) ensures reliable access.
- The worker handles heavy video/audio processing and exposes a `/transcribe` API endpoint for the main backend to use.

---

## Prerequisites
- Python 3.8+ (Python 3.11+ recommended)
- pip (Python package manager)
- ffmpeg (for audio/video processing)
- yt-dlp (for video downloading)
- instaloader (for Instagram carousel extraction)
- requests (for image downloading)
- OpenAI API key (for Whisper transcription and Vision API)
- ngrok account (free or paid)

---

## Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone <your-worker-repo-url>
cd vibeknowing-worker
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg
- **macOS:**
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```
- **Windows:**
  - Download from https://ffmpeg.org/download.html
  - Add ffmpeg to your system PATH

### 4. Install yt-dlp
```bash
pip install yt-dlp
```
Or download the binary from https://github.com/yt-dlp/yt-dlp/releases and add it to your PATH.

### 5. Set Your OpenAI API Key
- Get your key from https://platform.openai.com/api-keys
- Set it as an environment variable:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
  (On Windows: use `set OPENAI_API_KEY=sk-...` in Command Prompt)

---

## Running the Worker

### 1. Start the Worker Service
```bash
uvicorn worker:app --host 0.0.0.0 --port 8000
```
- The worker will be available at `http://localhost:8000/health`
- Test with: `curl http://localhost:8000/health`

### 2. Expose the Worker with ngrok
- Sign up at https://ngrok.com and download ngrok for your OS.
- Authenticate ngrok (replace YOUR_TOKEN):
  ```bash
  ngrok config add-authtoken YOUR_TOKEN
  ```
- Start the tunnel:
  ```bash
  ngrok http 8000
  ```
- You’ll get a public URL like `https://abc123.ngrok-free.app`.
- The worker API will be at `https://abc123.ngrok-free.app/transcribe`

### 3. Update Backend WORKER_URL
- In your backend’s environment variables, set:
  ```bash
  WORKER_URL=https://abc123.ngrok-free.app/transcribe
  ```
- Restart your backend if needed.

---

## Testing the Worker

- **Local test:**
  ```bash
  curl http://localhost:8000/health
  ```
- **ngrok test:**
  ```bash
  curl https://abc123.ngrok-free.app/health
  ```
- **Transcription test:**
  Use the provided `test_worker.py` script or POST to `/transcribe` with a JSON body:
  ```bash
  curl -X POST https://abc123.ngrok-free.app/transcribe \
    -H "Content-Type: application/json" \
    -d '{"url": "https://youtube.com/watch?v=..."}'
  ```

---

## Troubleshooting
- **Worker not reachable?**
  - Make sure the worker is running and ngrok tunnel is active.
  - Check for firewall or router issues blocking port 8000.
- **yt-dlp or ffmpeg errors?**
  - Ensure both are installed and available in your PATH.
  - Test with `yt-dlp --version` and `ffmpeg -version`.
- **OpenAI API errors?**
  - Double-check your API key and account limits.
- **ngrok tunnel closes?**
  - Free ngrok tunnels may time out after 8 hours. Restart as needed or upgrade to a paid plan for persistent tunnels.
- **Backend not connecting?**
  - Double-check the WORKER_URL in your backend’s environment variables.

---

## Security & Best Practices
- Never share your OpenAI API key or ngrok tunnel URL publicly.
- Use a strong ngrok authtoken and consider restricting allowed IPs (ngrok paid feature).
- Monitor worker logs for errors or abuse.
- Restart the worker and ngrok if you change your network or IP address.
- For production, consider running the worker on a dedicated, always-on device (e.g., Raspberry Pi, home server, or cloud VM with residential proxy).

---

## Useful Commands
- Check worker logs: `tail -f worker.log` (if logging enabled)
- Test endpoint: `curl http://localhost:8000/health`
- Test ngrok: `curl https://abc123.ngrok-free.app/health`
- Test transcription: see `test_worker.py` or use curl as above

---

## Requirements (Summary)
- Python 3.8+
- ffmpeg (for audio processing)
- yt-dlp (for video downloading)
- OpenAI API key
