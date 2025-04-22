import os
import sys
import time
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchaudio
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import re
from threading import Lock
import json

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
app = FastAPI(title="IndexTTS Master API")

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

# Load slave configuration
SLAVE_CONFIG = os.environ.get("SLAVE_CONFIG", '[{"url": "http://192.168.68.118:8003"}]')
try:
    SLAVES = json.loads(SLAVE_CONFIG)
except json.JSONDecodeError:
    logger.error("Invalid SLAVE_CONFIG format, using default")
    SLAVES = [{"url": "http://192.168.68.118:8003"}]

# Health check endpoint
@app.get("/health")
async def health():
    with inference_lock:
        return {"status": "healthy", "active_inferences": active_inferences, "role": "master"}

# Check slave availability
async def check_slaves():
    available_slaves = []
    for slave in SLAVES:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{slave['url']}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data["active_inferences"] < 1:
                            available_slaves.append(slave)
                            logger.info(f"Slave {slave['url']} available, load: {data['active_inferences']}")
                        else:
                            logger.warning(f"Slave {slave['url']} busy")
                    else:
                        logger.warning(f"Slave {slave['url']} health check failed")
        except Exception as e:
            logger.error(f"Failed to connect to slave {slave['url']}: {str(e)}")
    return available_slaves

# Split text into parts
def split_text(text, num_parts):
    if num_parts <= 1 or not text:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return [text]
    
    part_size = max(1, len(sentences) // num_parts)
    parts = []
    for i in range(0, len(sentences), part_size):
        part = " ".join(sentences[i:i + part_size]).strip()
        if part:
            parts.append(part)
    
    while len(parts) < num_parts:
        parts.append("")
    
    return parts[:num_parts]

# Concatenate audio files
def concatenate_audio_files(file_paths, output_path):
    if not file_paths:
        return None
    
    waveforms = []
    sample_rate = None
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            waveform, sr = torchaudio.load(file_path)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if sample rates differ
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            waveforms.append(waveform)
    
    if not waveforms:
        return None
    
    # Concatenate along time dimension
    concatenated = torch.cat(waveforms, dim=1)
    
    # Save concatenated file
    torchaudio.save(output_path, concatenated, sample_rate)
    logger.info(f"Concatenated audio saved to: {output_path}")
    return output_path

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

# Async inference request to slave
async def send_inference_request(session, url, voice, text, endpoint="/infer"):
    if not text:
        return None
    payload = {"voice": voice, "text": text} if endpoint == "/infer" else {"model": "indextts", "input": text, "voice": voice}
    try:
        async with session.post(f"{url}{endpoint}", json=payload) as response:
            if response.status != 200:
                logger.error(f"Error from {url}{endpoint}: {response.status}")
                return None
            result = await response.json()
            logger.info(f"Slave {url} confirmed completion: {result}")
            return result.get("output_path")
    except Exception as e:
        logger.error(f"Request to {url}{endpoint} failed: {str(e)}")
        return None

# Wait for file with retries
async def wait_for_file(file_path, max_attempts=10, delay=1):
    for attempt in range(max_attempts):
        if os.path.exists(file_path):
            logger.info(f"File found: {file_path}")
            return True
        logger.debug(f"File {file_path} not found, attempt {attempt + 1}/{max_attempts}")
        await asyncio.sleep(delay)
    logger.error(f"File {file_path} not found after {max_attempts} attempts")
    return False

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
    voice = request.voice
    input_text = request.input
    next_input = request.next_input
    
    available_slaves = await check_slaves()
    num_workers = len(available_slaves) + 1
    logger.info(f"Available workers: {num_workers} (master + {len(available_slaves)} slaves)")
    
    text_parts = split_text(input_text, num_workers)
    next_parts = split_text(next_input, num_workers) if next_input else ["" for _ in range(num_workers)]
    logger.info(f"Input text split into {len(text_parts)} parts")
    if next_input:
        logger.info(f"Next input split into {len(next_parts)} parts")
    
    outputs = [None] * len(text_parts)
    next_outputs = [None] * len(next_parts) if next_input else []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        tasks.append(infer(voice, text_parts[0]))
        for i, slave in enumerate(available_slaves, 1):
            if i < len(text_parts):
                tasks.append(send_inference_request(session, slave["url"], voice, text_parts[i], "/audio/speech"))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {str(result)}")
            else:
                outputs[i] = result
    
    for i in range(len(outputs)):
        if outputs[i] is None and text_parts[i]:
            logger.warning(f"Reprocessing part {i} locally")
            outputs[i] = await infer(voice, text_parts[i])
    
    # Ensure all output files exist
    for output in outputs:
        if output:
            await wait_for_file(output)
    
    # Concatenate audio files
    valid_outputs = [output for output in outputs if output]
    if valid_outputs:
        concatenated_path = os.path.join(OUTPUT_PATH, f"spk_{int(time.time())}_concat.wav")
        concatenated_path = concatenate_audio_files(valid_outputs, concatenated_path)
    else:
        concatenated_path = None
    
    if next_input:
        async def generate_next_parts():
            async with aiohttp.ClientSession() as session:
                tasks = []
                tasks.append(infer(voice, next_parts[0]))
                for i, slave in enumerate(available_slaves, 1):
                    if i < len(next_parts):
                        tasks.append(send_inference_request(session, slave["url"], voice, next_parts[i], "/audio/speech"))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        next_outputs[i] = result
                
                for i in range(len(next_outputs)):
                    if next_outputs[i] is None and next_parts[i]:
                        next_outputs[i] = await infer(voice, next_parts[i])
                
                # Ensure next output files exist
                for output in next_outputs:
                    if output:
                        await wait_for_file(output)
                
                # Concatenate next audio files
                valid_next_outputs = [output for output in next_outputs if output]
                if valid_next_outputs:
                    next_concatenated_path = os.path.join(OUTPUT_PATH, f"spk_{int(time.time())}_concat.wav")
                    concatenate_audio_files(valid_next_outputs, next_concatenated_path)
        
        background_tasks.add_task(generate_next_parts)
    
    if concatenated_path and os.path.exists(concatenated_path):
        return FileResponse(concatenated_path, media_type="audio/wav", filename=os.path.basename(concatenated_path))
    elif outputs[0]:
        return FileResponse(outputs[0], media_type="audio/wav", filename=os.path.basename(outputs[0]))
    else:
        raise HTTPException(status_code=500, detail="No audio generated")

# /generate_and_play endpoint
@app.post("/generate_and_play")
async def generate_and_play(request: TTSRequest):
    voice = request.voice
    text = request.text
    
    available_slaves = await check_slaves()
    num_workers = len(available_slaves) + 1
    logger.info(f"Available workers: {num_workers} (master + {len(available_slaves)} slaves)")
    
    text_parts = split_text(text, num_workers)
    logger.info(f"Text split into {len(text_parts)} parts")
    
    outputs = [None] * len(text_parts)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        tasks.append(infer(voice, text_parts[0]))
        for i, slave in enumerate(available_slaves, 1):
            if i < len(text_parts):
                tasks.append(send_inference_request(session, slave["url"], voice, text_parts[i]))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {str(result)}")
            else:
                outputs[i] = result
    
    for i in range(len(outputs)):
        if outputs[i] is None and text_parts[i]:
            logger.warning(f"Reprocessing part {i} locally")
            outputs[i] = await infer(voice, text_parts[i])
    
    for output in outputs:
        if output:
            await wait_for_file(output)
    
    valid_outputs = [output for output in outputs if output]
    if valid_outputs:
        concatenated_path = os.path.join(OUTPUT_PATH, f"spk_{int(time.time())}_concat.wav")
        concatenated_path = concatenate_audio_files(valid_outputs, concatenated_path)
        return {"output": concatenated_path}
    return {"outputs": valid_outputs}

# Startup event
@app.on_event("startup")
async def startup_event():
    available_slaves = await check_slaves()
    logger.info(f"Found {len(available_slaves)} available slaves")

# Main execution
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
