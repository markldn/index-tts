import os
import sys
import time
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import torchaudio
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from threading import Lock

# Add paths for indextts module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize i18n
i18n = I18nAuto(language="zh_CN")
MODE = 'local'

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize TTS model
tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=device)
logger.info("IndexTTS initialized")

# Get output path from environment
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("voices", exist_ok=True)

# Voice cache
voice_cache = {}

# Thread pool with single worker
executor = ThreadPoolExecutor(max_workers=1)

# FastAPI setup
app = FastAPI(title="IndexTTS Slave API")

# Pydantic models
class TTSRequest(BaseModel):
    voice: str
    text: str

class TTSAudioRequest(BaseModel):
    model: str
    input: str
    voice: str
    next_input: str = None

# Track active inferences
active_inferences = 0
inference_lock = Lock()

# Health check endpoint
@app.get("/health")
async def health():
    with inference_lock:
        return {"status": "healthy", "active_inferences": active_inferences, "role": "slave"}

# Synchronous TTS inference
def infer_sync(voice, text, output_path=None):
    global active_inferences
    with inference_lock:
        active_inferences += 1
    
    try:
        if not text:
            return None
        if not output_path:
            output_path = os.path.join(OUTPUT_PATH, f"spk_{int(time.time())}_{os.getpid()}.wav")
        
        if not os.path.exists(voice):
            raise ValueError(f"Voice file {voice} not found")
        
        logger.info(f"Generating audio for text: {text[:50]}...")
        start_total = time.time()
        
        if voice not in voice_cache:
            logger.info("Caching voice file")
            voice_cache[voice] = True
        
        with torch.no_grad():
            tts.infer(audio_prompt=voice, text=text, output_path=output_path, device=device)
        
        logger.info(f"Total inference took {time.time() - start_total:.3f} seconds")
        
        if device.type == "mps":
            torch.mps.empty_cache()
            logger.debug(f"MPS memory allocated: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
        
        return output_path
    finally:
        with inference_lock:
            active_inferences -= 1

# Async inference wrapper
async def infer(voice, text, output_path=None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, infer_sync, voice, text, output_path)

# /infer endpoint
@app.post("/infer")
async def api_infer(request: TTSRequest):
    try:
        output_path = await infer(request.voice, request.text)
        return {"output_path": output_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# /audio/speech endpoint
@app.post("/audio/speech")
async def audio_speech(request: TTSAudioRequest, background_tasks: BackgroundTasks):
    try:
        output_path = await infer(request.voice, request.input)
        if not output_path or not os.path.exists(output_path):
            raise ValueError(f"Output file {output_path} not found")
        
        if request.next_input:
            async def generate_next_audio():
                await infer(request.voice, request.next_input)
            background_tasks.add_task(generate_next_audio)
        
        return {"output_path": output_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Main execution
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
