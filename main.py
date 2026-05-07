import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk

# Importowanie naszych nowych modułów
from training_data import TrainingData
from vision_processor import VisionProcessor, mp_pose
from audio_engine import AudioEngine


class PersonalTrainerApp:
    """Główna klasa GUI (Widok i Kontroler logiczny dla treningu)."""

    def __init__(self, window):
        self.window = window
        self.window.title("Trener Personalny")
        self.window.geometry("1100x750")
        self.window.configure(bg="#2B2B2B")

        # Inicjalizacja modułów
        self.data_manager = TrainingData()
        self.vision = VisionProcessor()
        self.audio = AudioEngine(self.window, self.start_training, self.stop_training)

        # Zmienne treningowe
        self.counter = 0
        self.stage = None
        self.is_running = False

        # Kalibracja
        self.calibration_reps = 3
        self.calibration_count = 0
        self.calibration_angles = []
        self.calibration_done = False
        self.target_depth = 90.0
        self.current_min_angle = 180.0

        # Zmienne wizyjne
        self.cap = None
        self.pose = None

        self._setup_ui()

    def _setup_ui(self):
        BG_COLOR = "#2B2B2B"
        PANEL_COLOR = "#333333"
        TEXT_COLOR = "#FFFFFF"
        ACCENT_COLOR = "#4CAF50"
        DANGER_COLOR = "#F44336"
        INFO_COLOR = "#2196F3"
        FONT_TITLE = ("Segoe UI", 20, "bold")
        FONT_NORMAL = ("Segoe UI", 12)
        FONT_BTN = ("Segoe UI", 12, "bold")

        top_frame = tk.Frame(self.window, bg=BG_COLOR)
        top_frame.pack(fill=tk.X, pady=(20, 10))

        tk.Label(top_frame, text="SYSTEM ANALIZY PRZYSIADÓW SUMO",
                 font=FONT_TITLE, bg=BG_COLOR, fg=ACCENT_COLOR).pack()

        self.status_label = tk.Label(top_frame, text="Oczekuję na komendę głosową 'start' lub 'stop'",
                                     font=FONT_NORMAL, bg=BG_COLOR, fg="#AAAAAA")
        self.status_label.pack(pady=5)

        main_frame = tk.Frame(self.window, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left_panel = tk.Frame(main_frame, bg=PANEL_COLOR, bd=0,
                              highlightbackground="#444444", highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, ipadx=20, ipady=20)

        tk.Label(left_panel, text="PANEL STEROWANIA", font=("Segoe UI", 14, "bold"),
                 bg=PANEL_COLOR, fg=TEXT_COLOR).pack(pady=(20, 30))

        tk.Button(left_panel, text="▶ START TRENINGU", command=self.start_training,
                  bg=ACCENT_COLOR, fg=TEXT_COLOR, font=FONT_BTN, width=20, height=2,
                  relief="flat", cursor="hand2", activebackground="#45a049").pack(pady=10, padx=20)

        tk.Button(left_panel, text="⏹ STOP / ZAPISZ", command=self.stop_training,
                  bg=DANGER_COLOR, fg=TEXT_COLOR, font=FONT_BTN, width=20, height=2,
                  relief="flat", cursor="hand2", activebackground="#da190b").pack(pady=10, padx=20)

        tk.Button(left_panel, text="POKAŻ STATYSTYKI", command=self.data_manager.show_stats,
                  bg=INFO_COLOR, fg=TEXT_COLOR, font=FONT_BTN, width=20, height=2,
                  relief="flat", cursor="hand2", activebackground="#0b7dda").pack(pady=10, padx=20)

        right_panel = tk.Frame(main_frame, bg="#000000", bd=2, relief="flat")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))

        self.video_label = tk.Label(right_panel, bg="#000000", text="KAMERA WYŁĄCZONA",
                                    font=("Segoe UI", 16), fg="#555555")
        self.video_label.pack(expand=True)

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
        self.calibration_count = 0
        self.calibration_angles = []
        self.calibration_done = False
        self.current_min_angle = 180.0

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.status_label.config(text="Trening trwa...", fg="green")
        self.audio.speak("Trening rozpoczęty. Pierwsze trzy powtórzenia to kalibracja.")
        self.process_video()

    def stop_training(self):
        if not self.is_running:
            return

        self.is_running = False
        self.status_label.config(text="Trening zatrzymany.", fg="red")

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.pose:
            self.pose.close()
            self.pose = None

        self.data_manager.save_to_csv(self.counter)
        msg = self.data_manager.get_progression_message(self.counter)
        self.audio.speak(msg)

        messagebox.showinfo("Zapisano", f"Trening zapisany w pliku: {self.data_manager.history_file}")

    def process_video(self):
        if not self.is_running or self.cap is None or self.pose is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.window.after(10, self.process_video)
            return

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image = frame.copy()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            self._handle_pose_logic(image, lm, results)

        image_rgb_to_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(image_rgb_to_show))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if self.is_running:
            self.window.after(10, self.process_video)

    def _handle_pose_logic(self, image, lm, results):
        left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        right_ankle_x = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
        left_ankle_x = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
        right_hip_x = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x
        left_hip_x = lm[mp_pose.PoseLandmark.LEFT_HIP.value].x

        angle = self.vision.calculate_angle(left_hip, left_knee, left_ankle)
        is_sumo = abs(left_ankle_x - right_ankle_x) > abs(left_hip_x - right_hip_x) * 1.5
        ar_color = (0, 255, 0)

        # Logika Kalibracji
        if not self.calibration_done:
            cv2.putText(image, f"FAZA KALIBRACJI: {self.calibration_count + 1}/{self.calibration_reps}",
                        (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)
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
                self.audio.speak(str(self.calibration_count))

                if self.calibration_count >= self.calibration_reps:
                    self.target_depth = sum(self.calibration_angles) / len(self.calibration_angles) + 10
                    self.calibration_done = True
                    self.stage = None
                    self.audio.speak(f"Kalibracja zakonczona. Docelowy kat to {int(self.target_depth)} stopni.")

        # Logika Treningu Właściwego
        else:
            cv2.putText(image, f"CEL: {int(self.target_depth)} stopni",
                        (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
            if not is_sumo:
                ar_color = (0, 0, 255)
                cv2.putText(image, "SZERZEJ STOPY!", (10, 140), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

            if angle > 160:
                self.stage = "gora"
            if angle < self.target_depth and self.stage == "gora":
                if is_sumo:
                    self.stage = "dol"
                    self.counter += 1
                    ar_color = (0, 255, 0)
                    self.audio.speak(str(self.counter))
                else:
                    self.stage = "blad"
                    self.audio.speak("Szerzej stopy")

        self.vision.draw_protractor(image, left_hip, left_knee, left_ankle, angle, ar_color)
        self.vision.draw_landmarks(image, results.pose_landmarks)
        cv2.putText(image, f"Powt: {self.counter}", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(image, f"Kat: {int(angle)}", (10, 190), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)


if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalTrainerApp(root)
    root.mainloop()