from fastapi import FastAPI
import subprocess
import sys
import os

app = FastAPI()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
process = None

@app.post("/analyze_video")
def start_proctoring(video_url: str = "0"):
    global process
    # Stop existing process if running
    if process and process.poll() is None:
        process.terminate()

    source = video_url if video_url else "0"
    script_path = os.path.join(PROJECT_DIR, "proctor.py")
    process = subprocess.Popen(
        [sys.executable, script_path, source],
        cwd=PROJECT_DIR
    )
    return {"message": "Proctoring started!", "source": source, "pid": process.pid}

@app.post("/stop")
def stop_proctoring():
    global process
    if process and process.poll() is None:
        process.terminate()
        return {"message": "Proctoring stopped!"}
    return {"message": "Nothing was running"}