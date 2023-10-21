import queue
import re
import sys
import threading
import time
from typing import Dict, List

import numpy as np
import pyaudio
import requests


STEP_IN_SEC: int = 1    
LENGTH_IN_SEC: int = 6    
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE
MAX_SENTENCE_CHARACTERS = 80
SERVER_API_ENDPOINT = "http://localhost:8000/transcribe"

audio_queue = queue.Queue()
length_queue = queue.Queue(maxsize=LENGTH_IN_SEC)


def server(audio_data) -> str:
    response = requests.post(SERVER_API_ENDPOINT,
                             data=audio_data,
                             headers={'Content-Type': 'application/octet-stream'})
    result = response.json()
    return result["text"]


def producer_thread():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,    
    )

    print("Transcription")
    
    while True:
        audio_data = b""
        for _ in range(STEP_IN_SEC):
            chunk = stream.read(RATE)    
            audio_data += chunk

        audio_queue.put(audio_data)   

def consumer_thread(stats):
    while True:
        if length_queue.qsize() >= LENGTH_IN_SEC:
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

        transcription_to_visualize = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

        transcription_postprocessing_end_time = time.time()

        sys.stdout.write('\033[K' + transcription_to_visualize + '\r')

        audio_queue.task_done()

        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)


if __name__ == "__main__":
    stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}

    producer = threading.Thread(target=producer_thread)
    producer.start()

    consumer = threading.Thread(target=consumer_thread, args=(stats,))
    consumer.start()

    try:
        producer.join()
        consumer.join()
    except KeyboardInterrupt:
        print("Exiting...")
        print("Number of processed chunks: ", len(stats["overall"]))
        print(f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s")
        print(
            f"Transcription time: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
        )
        print(
            f"Postprocessing time: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
        )
        print(f"The average latency is {np.mean(stats['overall'])+STEP_IN_SEC:.4f}s")
