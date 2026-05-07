import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pyttsx3
import csv
import os
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import threading
import queue
import json
import sounddevice as sd
import vosk
import pyttsx3
import winsound

# --- POPRAWKA IMPORTU MEDIAPIPE ---
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError):
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

class PersonalTrainerApp:
    def __init__(self, window):
        self.last_spoken_text = ""
        self.last_spoken_time = 0
        self.engine = pyttsx3.init()
        self.tts_queue = queue.Queue()

        self.window = window
        self.window.after(100, self.process_tts_queue)
        self.window.title("Trener Personalny")
        self.window.geometry("1100x750")
        self.window.configure(bg="#2B2B2B")

        # Zmienne treningowe
        self.counter = 0                   # licznik właściwego treningu
        self.stage = None
        self.is_running = False
        self.history_file = "treningi_historia.csv"

        # Kalibracja
        self.calibration_reps = 3
        self.calibration_count = 0         # osobny licznik kalibracji
        self.calibration_angles = []
        self.calibration_done = False
        self.target_depth = 90.0
        self.current_min_angle = 180.0

        # Audio / Vosk
        self.audio_queue = queue.Queue()

        # Kamera
        self.cap = None
        self.pose = None

        # =========================
        # DESIGN & LAYOUT
        # =========================
        # Style constants
        BG_COLOR = "#2B2B2B"
        PANEL_COLOR = "#333333"
        TEXT_COLOR = "#FFFFFF"
        ACCENT_COLOR = "#4CAF50"  # Green
        DANGER_COLOR = "#F44336"  # Red
        INFO_COLOR = "#2196F3"  # Blue
        FONT_TITLE = ("Segoe UI", 20, "bold")
        FONT_NORMAL = ("Segoe UI", 12)
        FONT_BTN = ("Segoe UI", 12, "bold")

        # Top Bar
        top_frame = tk.Frame(window, bg=BG_COLOR)
        top_frame.pack(fill=tk.X, pady=(20, 10))

        self.label = tk.Label(
            top_frame,
            text="SYSTEM ANALIZY PRZYSIADÓW SUMO",
            font=FONT_TITLE,
            bg=BG_COLOR,
            fg=ACCENT_COLOR
        )
        self.label.pack()

        self.status_label = tk.Label(
            top_frame,
            text="Oczekuję na komendę głosową 'start' lub 'stop'",
            font=FONT_NORMAL,
            bg=BG_COLOR,
            fg="#AAAAAA"
        )
        self.status_label.pack(pady=5)

        # Main Content Container
        main_frame = tk.Frame(window, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Panel (Controls)
        left_panel = tk.Frame(main_frame, bg=PANEL_COLOR, bd=0, highlightbackground="#444444", highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, ipadx=20, ipady=20)

        controls_label = tk.Label(left_panel, text="PANEL STEROWANIA", font=("Segoe UI", 14, "bold"), bg=PANEL_COLOR, fg=TEXT_COLOR)
        controls_label.pack(pady=(20, 30))

        tk.Button(
            left_panel,
            text="▶ START TRENINGU",
            command=self.start_training,
            bg=ACCENT_COLOR,
            fg=TEXT_COLOR,
            font=FONT_BTN,
            width=20,
            height=2,
            relief="flat",
            cursor="hand2",
            activebackground="#45a049",
            activeforeground="white"
        ).pack(pady=10, padx=20)

        tk.Button(
            left_panel,
            text="⏹ STOP / ZAPISZ",
            command=self.stop_training,
            bg=DANGER_COLOR,
            fg=TEXT_COLOR,
            font=FONT_BTN,
            width=20,
            height=2,
            relief="flat",
            cursor="hand2",
            activebackground="#da190b",
            activeforeground="white"
        ).pack(pady=10, padx=20)

        tk.Button(
            left_panel,
            text="POKAŻ STATYSTYKI",
            command=self.show_stats,
            bg=INFO_COLOR,
            fg=TEXT_COLOR,
            font=FONT_BTN,
            width=20,
            height=2,
            relief="flat",
            cursor="hand2",
            activebackground="#0b7dda",
            activeforeground="white"
        ).pack(pady=10, padx=20)

        # Right Panel (Video)
        right_panel = tk.Frame(main_frame, bg="#000000", bd=2, relief="flat")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

        self.video_label = tk.Label(right_panel, bg="#000000", text="KAMERA WYŁĄCZONA", font=("Segoe UI", 16), fg="#555555")
        self.video_label.pack(expand=True)

        # Wątek głosowy
        self.voice_thread = threading.Thread(target=self.voice_listener, daemon=True)
        self.voice_thread.start()
    # =========================
    # GŁOS / VOSK
    # =========================
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_queue.put(bytes(indata))

    def voice_listener(self):
        """Nasłuchiwanie komend 'start' / 'stop' przez Vosk."""
        if not os.path.exists("model"):
            print("Brak folderu 'model' z modelem Vosk. Sterowanie głosowe wyłączone.")
            return

        try:
            model = vosk.Model("model")
            rec = vosk.KaldiRecognizer(model, 16000)

            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype='int16',
                channels=1,
                callback=self.audio_callback
            ):
                while True:
                    data = self.audio_queue.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text = res.get("text", "").lower()
                        if not text:
                            continue
                        print("Rozpoznano:", text)

                        if "start" in text and not self.is_running:
                            self.window.after(0, self.start_training)
                        elif "stop" in text and self.is_running:
                            self.window.after(0, self.stop_training)

        except Exception as e:
            print("Błąd Vosk:", e)

    # =========================
    # MATEMATYKA / RYSOWANIE
    # =========================
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def draw_protractor(self, image, a, b, c, angle, color):
        h, w, _ = image.shape

        pa = (int(a[0] * w), int(a[1] * h))
        pb = (int(b[0] * w), int(b[1] * h))
        pc = (int(c[0] * w), int(c[1] * h))

        cv2.line(image, pa, pb, color, 4)
        cv2.line(image, pb, pc, color, 4)
        cv2.circle(image, pb, 15, color, -1)
        cv2.circle(image, pb, 20, (255, 255, 255), 2)
        cv2.putText(
            image,
            str(int(angle)),
            (pb[0] + 20, pb[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    # =========================
    # TTS / ZAPIS / STATYSTYKI
    # =========================
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

        # dla numerów powtórzeń - krótki dźwięk zamiast TTS
        if text.isdigit():
            try:
                winsound.Beep(1000, 200)
            except Exception as e:
                print("Błąd Beep:", e)
            return

        self.tts_queue.put(text)
    def save_to_csv(self, count):
        file_exists = os.path.isfile(self.history_file)

        with open(self.history_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Data", "Powtorzenia"])
            writer.writerow([str(np.datetime64("now")), count])

    def check_progression(self):
        if not os.path.isfile(self.history_file):
            return

        counts = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    counts.append(int(row["Powtorzenia"]))
                except:
                    pass

        if len(counts) >= 3 and all(c >= 10 for c in counts[-3:]):
            self.speak("Świetna robota. Osiągnąłeś stabilne 10 powtórzeń. Twój nowy cel to 12.")
        else:
            self.speak(f"Trening zapisany. Wykonałeś {self.counter} powtórzeń.")

    def show_stats(self):
        if not os.path.isfile(self.history_file):
            messagebox.showwarning("Brak", "Brak danych.")
            return

        counts = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    counts.append(int(row["Powtorzenia"]))
                except:
                    pass

        if not counts:
            messagebox.showwarning("Brak", "Brak poprawnych danych do wykresu.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(counts, marker="o")
        plt.title("Postępy treningowe")
        plt.xlabel("Numer treningu")
        plt.ylabel("Liczba powtórzeń")
        plt.grid(True)
        plt.show()

    # =========================
    # START / STOP
    # =========================
    def start_training(self):
        if self.is_running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Błąd", "Nie udało się otworzyć kamery.")
            return

        self.is_running = True
        self.counter = 0
        self.stage = None

        # reset kalibracji
        self.calibration_count = 0
        self.calibration_angles = []
        self.calibration_done = False
        self.current_min_angle = 180.0

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.status_label.config(text="Trening trwa...", fg="green")
        self.speak("Trening rozpoczęty. Pierwsze trzy powtórzenia to kalibracja.")

        self.process_video()

    def stop_training(self):
        if not self.is_running:
            return

        self.is_running = False
        self.status_label.config(text="Trening zatrzymany.", fg="red")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.pose is not None:
            self.pose.close()
            self.pose = None

        self.save_to_csv(self.counter)
        self.check_progression()

        messagebox.showinfo("Zapisano", f"Trening zapisany w pliku: {self.history_file}")

    # =========================
    # GŁÓWNA PĘTLA WIDEO
    # =========================
    def process_video(self):
        if not self.is_running or self.cap is None or self.pose is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.is_running:
                self.window.after(10, self.process_video)
            return

        # Odbicie lustrzane
        frame = cv2.flip(frame, 1)

        # MediaPipe pracuje na RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        # Do rysowania używamy kopii w BGR
        image = frame.copy()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_ankle_x = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
            left_ankle_x = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
            right_hip_x = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x
            left_hip_x = lm[mp_pose.PoseLandmark.LEFT_HIP.value].x

            angle = self.calculate_angle(left_hip, left_knee, left_ankle)

            ankle_dist = abs(left_ankle_x - right_ankle_x)
            hip_dist = abs(left_hip_x - right_hip_x)
            is_sumo = ankle_dist > hip_dist * 1.5

            ar_color = (0, 255, 0)

            # ===== KALIBRACJA =====
            if not self.calibration_done:
                cv2.putText(
                    image,
                    f"FAZA KALIBRACJI: {self.calibration_count + 1}/{self.calibration_reps}",
                    (10, 100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 165, 255),
                    2
                )
                ar_color = (0, 165, 255)

                if angle < self.current_min_angle:
                    self.current_min_angle = angle

                if angle > 160:
                    self.stage = "gora"

                if angle < 100 and self.stage == "gora" and is_sumo:
                    self.stage = "dol"

                if angle > 150 and self.stage == "dol":
                    self.stage = "gora"
                    self.calibration_angles.append(self.current_min_angle)
                    self.calibration_count += 1
                    self.current_min_angle = 180.0
                    self.speak(str(self.calibration_count))

                    if self.calibration_count >= self.calibration_reps:
                        self.target_depth = sum(self.calibration_angles) / len(self.calibration_angles) + 10
                        self.calibration_done = True
                        self.stage = None
                        self.speak(
                            f"Kalibracja zakończona. Twój docelowy kąt to {int(self.target_depth)} stopni."
                        )

            # ===== TRENING WŁAŚCIWY =====
            else:
                cv2.putText(
                    image,
                    f"CEL: {int(self.target_depth)} stopni",
                    (10, 100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

                if not is_sumo:
                    ar_color = (0, 0, 255)
                    cv2.putText(
                        image,
                        "SZERZEJ STOPY!",
                        (10, 140),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                if angle > 160:
                    self.stage = "gora"

                if angle < self.target_depth and self.stage == "gora":
                    if is_sumo:
                        self.stage = "dol"
                        self.counter += 1
                        ar_color = (0, 255, 0)
                        self.speak(str(self.counter))
                    else:
                        self.stage = "blad"
                        self.speak("Szerzej stopy")

            # Rysowanie AR
            self.draw_protractor(image, left_hip, left_knee, left_ankle, angle, ar_color)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            cv2.putText(
                image,
                f"Powt: {self.counter}",
                (10, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 0, 255),
                2
            )

            cv2.putText(
                image,
                f"Kat: {int(angle)}",
                (10, 190),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Konwersja do Tkinter
        image_rgb_to_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb_to_show)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Następna klatka
        if self.is_running:
            self.window.after(10, self.process_video)


if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalTrainerApp(root)
    root.mainloop()

