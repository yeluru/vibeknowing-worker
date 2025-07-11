from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import tempfile
import glob
import os
from openai import OpenAI
import shutil
import math
import time
from typing import List

app = FastAPI()

# Replace with your OpenAI API key
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY, timeout=300.0)  # 5 minute timeout

class VideoRequest(BaseModel):
    url: str

def split_audio_ffmpeg(audio_file: str, max_size: int = 20*1024*1024) -> List[str]:
    """Split audio file into chunks smaller than max_size bytes"""
    file_size = os.path.getsize(audio_file)
    if file_size <= max_size:
        return [audio_file]
    
    # Get duration in seconds
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
         'default=noprint_wrappers=1:nokey=1', audio_file],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Error getting duration: {result.stderr}")
        return [audio_file]
    
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        print(f"Could not parse duration: {result.stdout}")
        return [audio_file]
    
    # Calculate number of chunks needed
    num_chunks = math.ceil(file_size / max_size)
    chunk_duration = duration / num_chunks
    
    chunk_paths = []
    base_name = os.path.splitext(audio_file)[0]
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)
        chunk_path = f"{base_name}_chunk_{i:03d}.mp3"
        
        cmd = [
            'ffmpeg', '-y', '-i', audio_file,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-acodec', 'libmp3lame',
            '-ar', '16000',
            '-ac', '1',
            '-b:a', '128k',
            chunk_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(chunk_path):
            chunk_paths.append(chunk_path)
        else:
            print(f"Error creating chunk {i}: {result.stderr}")
    
    return chunk_paths if chunk_paths else [audio_file]

def transcribe_with_retry(audio_file_path: str, max_retries: int = 3) -> str:
    """Transcribe audio file with retry logic"""
    for attempt in range(max_retries):
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return transcript
        except Exception as e:
            print(f"Transcription attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
    
    # This should never be reached, but just in case
    raise RuntimeError("All transcription attempts failed")

@app.post("/transcribe")
def transcribe_video(req: VideoRequest):
    url = req.url
    with tempfile.TemporaryDirectory() as temp_dir:
        # Try to get subtitles first
        cmd = [
            "yt-dlp",
            "--write-sub",
            "--write-auto-sub",
            "--sub-langs", "en,en-US,en-GB",
            "--sub-format", "vtt",
            "--skip-download",
            "--no-warnings",
            "-o", f"{temp_dir}/%(title)s.%(ext)s",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        vtt_files = glob.glob(f"{temp_dir}/*.vtt")
        if vtt_files:
            with open(vtt_files[0], "r", encoding="utf-8") as f:
                content = f.read()
            return {"method": "subtitles", "transcript": content}

        # Fallback: download audio and transcribe with OpenAI Whisper
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "192K",
            "--no-warnings",
            "-o", f"{temp_dir}/%(title)s.%(ext)s",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        audio_files = glob.glob(f"{temp_dir}/*.mp3")
        if audio_files:
            audio_file_path = audio_files[0]
            print(f"Processing audio file: {audio_file_path}")
            
            # Split audio if needed (20MB chunks for safety)
            max_size = 20 * 1024 * 1024
            chunk_paths = split_audio_ffmpeg(audio_file_path, max_size)
            print(f"Split into {len(chunk_paths)} chunks")
            
            full_transcript = ""
            for i, chunk_path in enumerate(chunk_paths):
                print(f"Transcribing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
                try:
                    transcript = transcribe_with_retry(chunk_path)
                    full_transcript += transcript + "\n"
                except Exception as e:
                    print(f"Failed to transcribe chunk {i+1}: {str(e)}")
                    # Continue with other chunks
                
                # Clean up chunk file if it's not the original
                if chunk_path != audio_file_path and os.path.exists(chunk_path):
                    os.remove(chunk_path)
            
            if full_transcript.strip():
                return {"method": "audio", "transcript": full_transcript.strip()}
            else:
                raise HTTPException(status_code=500, detail="Failed to transcribe any audio chunks")

        raise HTTPException(status_code=500, detail="Failed to get transcript from video.")

