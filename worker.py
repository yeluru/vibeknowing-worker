from fastapi import FastAPI, HTTPException, UploadFile, File
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

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "VibeKnowing Worker"}

# Get OpenAI API key from environment variable or hardcoded value
# PASTE YOUR API KEY HERE IF RUNNING LOCALLY WITHOUT ENV VARS
OPENAI_API_KEY_HARDCODED = ""

OPENAI_API_KEY = OPENAI_API_KEY_HARDCODED or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required. Please set OPENAI_API_KEY_HARDCODED in worker.py or use environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY, timeout=300.0)  # 5 minute timeout

import base64

class VideoRequest(BaseModel):
    url: str

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_frames(video_path, output_dir, interval=2.0):
    """Extract frames from video every `interval` seconds"""
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not available for frame extraction")
        return []

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps=1/{interval}',
        '-q:v', '2',  # High quality
        output_pattern
    ]
    
    print(f"Extracting frames: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error extracting frames: {result.stderr}")
        return []
        
    return sorted(glob.glob(os.path.join(output_dir, "*.jpg")))

def analyze_frames_with_vision(frame_paths, client):
    """Send frames to OpenAI Vision for text extraction"""
    if not frame_paths:
        return ""

    print(f"Analyzing {len(frame_paths)} frames with OpenAI Vision...")
    
    # Process in batches to avoid payload limits
    batch_size = 5
    full_text = []
    
    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "These are frames from a video. Extract all visible text from these frames. Ignore standard UI elements (like time, battery, app icons). Output the text sequentially as it appears. If text is repeated across frames, deduplicate it naturally. Return ONLY the text."}
                ]
            }
        ]
        
        for frame_path in batch:
            base64_image = encode_image(frame_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low" # Use low detail for text, usually sufficient and faster/cheaper
                }
            })
            
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Capable of vision
                messages=messages,
                max_tokens=1000
            )
            content = response.choices[0].message.content
            if content:
                full_text.append(content)
        except Exception as e:
            print(f"Vision API error: {e}")
            
    return "\n\n".join(full_text)

def split_audio_ffmpeg(audio_file: str, max_size: int = 20*1024*1024) -> List[str]:
    """Split audio file into chunks smaller than max_size bytes"""
    file_size = os.path.getsize(audio_file)
    if file_size <= max_size:
        return [audio_file]
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not available, returning original file")
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
    print(f"Starting transcription for URL: {url}")
    
    # Check if yt-dlp is available
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("yt-dlp not available")
        raise HTTPException(status_code=500, detail="yt-dlp not available on this system")
    
    try:
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
            print(f"Running subtitle command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(f"Subtitle command result: {result.returncode}")
            print(f"Subtitle command stdout: {result.stdout}")
            print(f"Subtitle command stderr: {result.stderr}")
            
            vtt_files = glob.glob(f"{temp_dir}/*.vtt")
            print(f"Found VTT files: {vtt_files}")
            
            if vtt_files:
                with open(vtt_files[0], "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Clean VTT content - extract only the actual text
                import re
                lines = content.splitlines()
                cleaned_lines = []
                seen_lines = set()
                
                for line in lines:
                    line = line.strip()
                    # Skip timestamps, headers, and empty lines
                    if (not line or 
                        line.startswith('WEBVTT') or 
                        '-->' in line or 
                        line.isdigit() or
                        line.startswith('Kind:') or
                        line.startswith('Language:') or
                        line.startswith('align:') or
                        line in seen_lines):
                        continue
                    
                    # Remove timestamp tags like <00:00:00.240><c> text </c>
                    # This regex removes all <...> tags
                    cleaned_line = re.sub(r'<[^>]+>', '', line)
                    cleaned_line = cleaned_line.strip()
                    
                    if cleaned_line and cleaned_line not in seen_lines:
                        cleaned_lines.append(cleaned_line)
                        seen_lines.add(cleaned_line)
                
                cleaned_content = " ".join(cleaned_lines)
                
                # Extract title using yt-dlp metadata (more reliable than filename)
                title = "Video"
                try:
                    title_cmd = ["yt-dlp", "--get-title", "--no-warnings", url]
                    title_result = subprocess.run(title_cmd, capture_output=True, text=True, timeout=10, cwd=temp_dir)
                    if title_result.returncode == 0 and title_result.stdout.strip():
                        title = title_result.stdout.strip()
                    else:
                        # Fallback: extract from VTT filename
                        vtt_filename = os.path.basename(vtt_files[0])
                        title = vtt_filename.replace('.en.vtt', '').replace('.vtt', '')
                except Exception as e:
                    print(f"Failed to extract title: {e}")
                    # Fallback: extract from VTT filename
                    vtt_filename = os.path.basename(vtt_files[0])
                    title = vtt_filename.replace('.en.vtt', '').replace('.vtt', '')
                
                print(f"Extracted title for {url}: {title}")
                return {"method": "subtitles", "transcript": cleaned_content, "title": title}

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
            print(f"Running audio command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            print(f"Audio command result: {result.returncode}")
            print(f"Audio command stdout: {result.stdout}")
            print(f"Audio command stderr: {result.stderr}")
            
            audio_files = glob.glob(f"{temp_dir}/*.mp3")
            print(f"Found audio files: {audio_files}")
            
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
                
                
                # Extract title using yt-dlp metadata
                title = "Video"
                try:
                    title_cmd = ["yt-dlp", "--get-title", "--no-warnings", url]
                    title_result = subprocess.run(title_cmd, capture_output=True, text=True, timeout=10, cwd=temp_dir)
                    if title_result.returncode == 0 and title_result.stdout.strip():
                        title = title_result.stdout.strip()
                    elif audio_files:
                        # Fallback: extract from filename (remove extension)
                        filename = os.path.basename(audio_files[0])
                        title = os.path.splitext(filename)[0]
                except Exception as e:
                    print(f"Failed to extract title: {e}")
                    if audio_files:
                        filename = os.path.basename(audio_files[0])
                        title = os.path.splitext(filename)[0]

                if full_transcript.strip():
                    return {"method": "audio", "transcript": full_transcript.strip(), "title": title}
                else:
                    raise HTTPException(status_code=500, detail="Failed to transcribe any audio chunks")

            # Fallback 4: Visual + Metadata Extraction
            print("Audio extraction failed. Attempting Visual + Metadata extraction...")
            try:
                # 0. Try Instaloader for Carousels (Multi-page posts)
                if "instagram.com" in url:
                    print("Checking for Instagram Carousel (Sidecar)...")
                    try:
                        import instaloader
                        L = instaloader.Instaloader()
                        # Extract shortcode
                        if "/p/" in url:
                            shortcode = url.split("/p/")[1].split("/")[0]
                        elif "/reel/" in url:
                            shortcode = url.split("/reel/")[1].split("/")[0]
                        else:
                            shortcode = None

                        if shortcode:
                            post = instaloader.Post.from_shortcode(L.context, shortcode)
                            print(f"Instaloader Post Type: {post.typename}")
                            
                            if post.typename == 'GraphSidecar':
                                print("Detected Instagram Sidecar (Carousel). Processing all slides...")
                                carousel_text = []
                                slide_count = 0
                                
                                for node in post.get_sidecar_nodes():
                                    slide_count += 1
                                    image_url = node.display_url
                                    if not image_url:
                                        continue
                                        
                                    print(f"Processing Slide {slide_count}: {image_url[:50]}...")
                                    
                                    # Download image to temp file
                                    try:
                                        import requests
                                        img_resp = requests.get(image_url)
                                        if img_resp.status_code == 200:
                                            slide_path = os.path.join(temp_dir, f"slide_{slide_count}.jpg")
                                            with open(slide_path, "wb") as f:
                                                f.write(img_resp.content)
                                            
                                            # Send to Vision
                                            slide_text = analyze_frames_with_vision([slide_path], client)
                                            if slide_text:
                                                carousel_text.append(f"Slide {slide_count}: {slide_text}")
                                    except Exception as img_err:
                                        print(f"Failed to process slide {slide_count}: {img_err}")

                                if carousel_text:
                                    full_carousel_transcript = "\n\n".join(carousel_text)
                                    # Add caption
                                    if post.caption:
                                        full_carousel_transcript = f"Caption: {post.caption}\n\n" + full_carousel_transcript
                                        
                                    return {
                                        "method": "instagram_carousel",
                                        "transcript": full_carousel_transcript,
                                        "title": f"Post by {post.owner_username}"
                                    }
                    except Exception as instaloader_err:
                        print(f"Instaloader check failed: {instaloader_err}")
                        # Continue to standard yt-dlp fallback

                # Download video AND metadata
                # Remove --skip-download to get the video file for vision processing
                dl_cmd = [
                    "yt-dlp", 
                    "--write-info-json", 
                    "--no-warnings",
                    "-o", f"{temp_dir}/%(title)s.%(ext)s",
                    url
                ]
                print(f"Downloading video for vision analysis: {' '.join(dl_cmd)}")
                dl_result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=120, cwd=temp_dir)
                
                # 1. Extract Metadata
                metadata_text = ""
                title = "Video"
                info_files = glob.glob(f"{temp_dir}/*.info.json")
                if info_files:
                    try:
                        import json
                        with open(info_files[0], 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Get title
                        title = metadata.get("title") or title
                        
                        # Get caption/description
                        if metadata.get("description"):
                            metadata_text = metadata["description"]
                        elif metadata.get("caption"):
                            metadata_text = metadata["caption"]
                        elif metadata.get("title") and metadata["title"] != "Video":
                            metadata_text = metadata["title"]
                            
                        print(f"Found metadata text: {len(metadata_text)} chars")
                    except Exception as e:
                        print(f"Error parsing metadata: {e}")

                # 2. Extract Visual Text (Vision)
                visual_text = ""
                video_files = glob.glob(f"{temp_dir}/*.mp4") # Instagram usually mp4
                if not video_files:
                     # Try other extensions
                     video_files = glob.glob(f"{temp_dir}/*.webm") + glob.glob(f"{temp_dir}/*.mkv")

                if video_files:
                    video_path = video_files[0]
                    print(f"Processing video for vision: {video_path}")
                    
                    # Extract frames (every 2 seconds)
                    frames_dir = os.path.join(temp_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    frame_paths = extract_frames(video_path, frames_dir, interval=2.0)
                    
                    if frame_paths:
                        # Analyze with OpenAI Vision
                        visual_text = analyze_frames_with_vision(frame_paths, client)
                        print(f"Extracted visual text: {len(visual_text)} chars")
                else:
                    print("No video file found for vision analysis.")

                # 3. Combine Results
                combined_transcript = ""
                if metadata_text:
                    combined_transcript += f"Caption: {metadata_text}\n\n"
                if visual_text:
                    combined_transcript += f"Visual Content:\n{visual_text}"
                
                if combined_transcript.strip():
                    return {
                        "method": "visual_and_metadata", 
                        "transcript": combined_transcript.strip(), 
                        "title": title
                    }

            except Exception as e:
                print(f"Visual/Metadata fallback failed: {e}")
                import traceback
                print(traceback.format_exc())

            raise HTTPException(status_code=500, detail="Failed to get transcript from video (No audio, subtitles, or visual content found).")
    except Exception as e:
        print(f"Error in transcribe_video: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe uploaded video or audio file"""
    print(f"Received file upload: {file.filename}, content_type: {file.content_type}")
    
    # Validate file type
    allowed_types = [
        'video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo',
        'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/m4a',
        'audio/x-m4a', 'audio/ogg', 'audio/webm'
    ]
    
    if file.content_type not in allowed_types:
        print(f"Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            file_ext = os.path.splitext(file.filename)[1] or '.mp4'
            temp_file_path = f"{temp_dir}/uploaded{file_ext}"
            
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            print(f"Saved file to: {temp_file_path}, size: {os.path.getsize(temp_file_path)} bytes")
            
            # Extract audio from video if needed
            audio_file_path = f"{temp_dir}/audio.mp3"
            cmd = [
                'ffmpeg', '-y', '-i', temp_file_path,
                '-vn',  # No video
                '-acodec', 'libmp3lame',
                '-ar', '16000',
                '-ac', '1',
                '-b:a', '128k',
                audio_file_path
            ]
            
            print(f"Extracting audio: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0 or not os.path.exists(audio_file_path):
                print(f"Audio extraction failed: {result.stderr}")
                raise HTTPException(status_code=500, detail="Failed to extract audio from file")
            
            print(f"Audio extracted: {audio_file_path}, size: {os.path.getsize(audio_file_path)} bytes")
            
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
                # Extract title from filename (remove extension)
                title = os.path.splitext(file.filename)[0]
                return {"method": "file_upload", "transcript": full_transcript.strip(), "title": title}
            else:
                raise HTTPException(status_code=500, detail="Failed to transcribe any audio chunks")
                
    except Exception as e:
        print(f"Error in transcribe_file: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
