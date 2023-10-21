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

