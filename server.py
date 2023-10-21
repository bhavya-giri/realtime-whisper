import asyncio
import numpy as np
from uvicorn import run
from fastapi import Depends,FastAPI,Request
from faster_whisper import WhisperModel

DEVICE = 'cpu'
COMPUTE_TYPE = 'int8'
NUM_WORKERS = 10
TYPE = 'base.en'
LANGUAGE = 'en'
THREADS = 4
BEAN_SIZE = 5
VAD_FILTER = True

model = WhisperModel(
                    TYPE,
                    device = DEVICE,
                    compute_type=COMPUTE_TYPE,
                    num_workers=NUM_WORKERS,
                    cpu_threads=THREADS,
                    download_root="./models"
        )

def model_transcribe(audio_array):
    segments,_ = model.transcibe(
                                   audio_array,
                                   language=LANGUAGE,
                                   beam_size=BEAN_SIZE,
                                   vad_filter=VAD_FILTER,
                                   vad_parameters=dict(min_silence_duration_ms=1000)
    )
    segments = [s.text for s in segments]
    transcription = " ".join(segments)
    transcription = transcription.strip()
    return transcription

app = FastAPI()

async def parse_body(request: Request):
    data: bytes = await request.body()
    return data

@app.post("/transcribe")
async def transcribe(audio_data: bytes = Depends(parse_body)):
    audio_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0
    try:
        text = await asyncio.get_running_loop().run_in_executor(None,model_transcribe,audio_array)
    except Exception as e:
        print(e)
    return {"text": text}

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
 
