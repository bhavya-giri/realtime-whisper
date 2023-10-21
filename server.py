import asyncio

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Request
from faster_whisper import WhisperModel
from uvicorn import run

app = FastAPI()

DEVICE = "cpu"
NUM_WORKERS = 10
MODEL = "base.en"
COMPUTE_TYPE="int8"
LANGUAGE_CODE = "en"
THREADS = 4
VAD_FILTER = True


def create_whisper_model() -> WhisperModel:
    whisper = WhisperModel(MODEL,
                           device=DEVICE,
                           compute_type=COMPUTE_TYPE,
                           num_workers=NUM_WORKERS,
                           cpu_threads=THREADS,
                           download_root="./models")
    print("Loaded model")
    return whisper


model = create_whisper_model()
print("Loaded model")


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data


def pipeline(model: WhisperModel, audio_data_array) -> str:
    segments, _ = model.transcribe(audio_data_array,
                                   language=LANGUAGE_CODE,
                                   beam_size=5,
                                   vad_filter=VAD_FILTER,
                                   vad_parameters=dict(min_silence_duration_ms=1000))
    segments = [s.text for s in segments]
    transcription = " ".join(segments)
    transcription = transcription.strip()
    return transcription


@app.post("/transcribe")
async def predict(audio_data: bytes = Depends(parse_body)):
    audio_data_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0
    try:
        text = await asyncio.get_running_loop().run_in_executor(None, pipeline, model,
                                                                  audio_data_array)
    except Exception as e:
        print(e)
        text = "Error"
    return {"text": text}


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
