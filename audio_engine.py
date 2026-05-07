import pyttsx3
import queue
import time
import threading
import json
import os
import sounddevice as sd
import vosk
import winsound


class AudioEngine:
    """Klasa odpowiedzialna za komunikację głosową (TTS) oraz rozpoznawanie mowy (STT)."""

    def __init__(self, tk_window, on_start_cmd, on_stop_cmd):
        self.window = tk_window
        self.on_start = on_start_cmd
        self.on_stop = on_stop_cmd

        self.engine = pyttsx3.init()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.last_spoken_text = ""
        self.last_spoken_time = 0

        # Uruchomienie pętli TTS i wątku STT
        self.window.after(100, self.process_tts_queue)
        self.voice_thread = threading.Thread(target=self.voice_listener, daemon=True)
        self.voice_thread.start()

    def process_tts_queue(self):
        if not self.tts_queue.empty():
            text = self.tts_queue.get()
            try:
                print("MÓWIĘ:", text)
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print("Błąd TTS:", e)
        self.window.after(100, self.process_tts_queue)

    def speak(self, text):
        now = time.time()
        text = str(text)

        if text == self.last_spoken_text and now - self.last_spoken_time < 1.5:
            return

        self.last_spoken_text = text
        self.last_spoken_time = now

        if text.isdigit():
            try:
                winsound.Beep(1000, 200)
            except Exception as e:
                print("Błąd Beep:", e)
            return

        self.tts_queue.put(text)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_queue.put(bytes(indata))

    def voice_listener(self):
        if not os.path.exists("model"):
            print("Brak folderu 'model' z modelem Vosk. Sterowanie głosowe wyłączone.")
            return

        try:
            model = vosk.Model("model")
            rec = vosk.KaldiRecognizer(model, 16000)

            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                   channels=1, callback=self.audio_callback):
                while True:
                    data = self.audio_queue.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text = res.get("text", "").lower()
                        if not text:
                            continue
                        print("Rozpoznano:", text)

                        if "start" in text:
                            self.window.after(0, self.on_start)
                        elif "stop" in text:
                            self.window.after(0, self.on_stop)
        except Exception as e:
            print("Błąd Vosk:", e)