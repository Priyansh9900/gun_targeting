# audio_detection.py
import numpy as np
import sounddevice as sd
import threading

clap_detected = False
clap_index = 0

def is_detected(path):
    global clap_detected, clap_index

    def callback(indata, frames, time, status):
        global clap_detected  # Declare clap_detected as a global variable

        if status:
            print(status)

        energy = np.sum(np.square(indata[:, 0]))

        if not clap_detected and energy > 5:
            print("CLAPPED!")
            clap_detected = True
            clap_index = len(path)

    sample_rate = sd.query_devices(None, 'input')['default_samplerate']

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
        print("Listening for a clap...")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            pass
