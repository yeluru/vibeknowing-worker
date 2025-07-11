import httpx

WORKER_URL = "https://bddbf892f7c4.ngrok-free.app/transcribe"

def get_transcript_from_worker(youtube_url: str):
    payload = {"url": youtube_url}
    try:
        response = httpx.post(WORKER_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data  # Contains 'method' and 'transcript'
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    url = input("Enter a YouTube video URL: ")
    result = get_transcript_from_worker(url)
    print(result)