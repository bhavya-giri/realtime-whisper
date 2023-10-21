import queue
import sys
import re
import threading
import time
from typing import Dict, List
import numpy as np
import pyaudio
import requests

STEP_SEC = 1
LENGTH_SEC = 6
CHANNELS = 1
RATE = CHUNKS = 16000
MAX_CHARS = 80
SERVER_ENDPOINT = "https://localhost:8000/transcribe"

audio_queue = queue.Queue()
length_queue = queue.Queue(maxsize=LENGTH_SEC)

def server(audio_data):
    response = requests.post(SERVER_ENDPOINT, data=audio_data, headers={'Content-Type': 'application/octet-stream'})
    result =response.json()
    return result["text"]

def producer():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format = pyaudio.paInt16,
        channels = CHANNELS,
        rate=RATE,
        input = True,
        frames_per_buffer=CHUNKS
    )
    
    while True:
        audio_data = b""
        for _ in range(STEP_SEC):
            chunk = stream.read(RATE)    
            audio_data += chunk

        audio_queue.put(audio_data)    

def consumer(stats):
    while True:
        if length_queue.qsize() >= LENGTH_SEC:
            with length_queue.mutex:
                length_queue.queue.clear()
                print()

        audio_data = audio_queue.get()
        transcription_start_time = time.time()
        length_queue.put(audio_data)
        audio_data_to_process = b""
        for i in range(length_queue.qsize()):
            audio_data_to_process += length_queue.queue[i]

        try:
            transcription = server(audio_data_to_process)
            transcription = re.sub(r"\[.*\]", "", transcription)
            transcription = re.sub(r"\(.*\)", "", transcription)
        except:
            transcription = "Error"

        transcription_end_time = time.time()
        transcription_to_visualize = transcription.ljust(MAX_CHARS, " ")

        transcription_postprocessing_end_time = time.time()

        sys.stdout.write('\033[K' + transcription_to_visualize + '\r')

        audio_queue.task_done()

        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)
    