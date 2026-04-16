import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pyttsx3
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# --- POPRAWKA IMPORTU MEDIAPIPE ---
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

    print("Sukces! Moduły MediaPipe załadowane bezpośrednio.")
except (ImportError, AttributeError):
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

    print("Sukces! Moduły MediaPipe załadowane przez solutions.")

# Inicjalizacja silnika mowy (wymagane!)
engine = pyttsx3.init()


class PersonalTrainerApp:
    def __init__(self, window):
        self.window = window
        self.window.title(f"Trener Personalny - Indeks: 255693")

        self.counter = 0
        self.stage = None
        self.is_running = False
        self.history_file = "treningi_historia.csv"

        # Interfejs TK-Inter [cite: 12]
        self.label = tk.Label(window, text="System Analizy Przysiadów Sumo", font=("Arial", 14))
        self.label.pack(pady=10)

        tk.Button(window, text="START TRENINGU", command=self.start_training, bg="green", fg="white", width=20).pack(
            pady=5)
        tk.Button(window, text="STOP / ZAPISZ", command=self.stop_training, bg="red", fg="white", width=20).pack(pady=5)
        tk.Button(window, text="POKAŻ STATYSTYKI", command=self.show_stats, width=20).pack(pady=5)

        self.video_label = tk.Label(window)
        self.video_label.pack()
        self.cap = None

    def calculate_angle(self, a, b, c):
        """Oblicza kąt między trzema punktami przy użyciu arctan2."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def speak(self, text):
        """Generuje wskazówki głosowe offline."""
        engine.say(text)
        engine.runAndWait()

    def save_to_csv(self, count):
        """Lokalna baza danych w pliku CSV."""
        file_exists = os.path.isfile(self.history_file)
        with open(self.history_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['Data', 'Powtorzenia'])
            writer.writerow([np.datetime64('now'), count])

    def show_stats(self):
        """Wizualizacja Matplotlib[cite: 10, 11]."""
        if not os.path.isfile(self.history_file):
            messagebox.showwarning("Brak danych", "Najpierw wykonaj trening!")
            return

        counts = []
        with open(self.history_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader: counts.append(int(row['Powtorzenia']))

        plt.figure(figsize=(8, 4))
        plt.plot(counts, marker='o', linestyle='-', color='b')
        plt.title("Postępy treningowe (Liczba powtórzeń)")
        plt.grid(True)
        plt.show()

    def start_training(self):
        self.is_running = True
        self.counter = 0
        self.cap = cv2.VideoCapture(0)
        self.speak("Rozpoczynamy. Rozstaw stopy szeroko do przysiadu sumo.")
        self.process_video()

    def stop_training(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.save_to_csv(self.counter)

        # Logika if-else dotycząca planowania [cite: 24, 25]
        if self.counter >= 10:
            self.speak(f"Doskonale! Zrobiłeś {self.counter} powtórzeń. Następny cel: {self.counter + 2}.")
        else:
            self.speak(f"Trening zakończony. Wynik to {self.counter}.")

        messagebox.showinfo("Zapisano", f"Trening zapisany w {self.history_file}")

    def process_video(self):
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret: break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    # Pobranie punktów (biodro, kolano, kostka)
                    hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    angle = self.calculate_angle(hip, knee, ankle)

                    # Logika Sumo: Sprawdź szerokość kostek względem bioder
                    ankle_dist = abs(
                        lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
                    hip_dist = abs(
                        lm[mp_pose.PoseLandmark.LEFT_HIP.value].x - lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x)

                    # Liczenie powtórzeń [cite: 17, 21]
                    if angle > 160: self.stage = "gora"
                    if angle < 90 and self.stage == "gora":
                        if ankle_dist > hip_dist * 1.5:  # Postprocessing matematyczny
                            self.stage = "dol"
                            self.counter += 1
                            self.speak(str(self.counter))
                        else:
                            self.speak("Rozstaw stopy szerzej")
                            self.stage = "blad"

                    # AR: Rzeczywistość rozszerzona na żywo [cite: 16, 17]
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(image, f"Powt: {self.counter}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 2)
                    cv2.circle(image, (int(knee[0] * frame.shape[1]), int(knee[1] * frame.shape[0])), 10, (0, 255, 0),
                               -1)

                img = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                self.window.update()


root = tk.Tk()
app = PersonalTrainerApp(root)
root.mainloop()