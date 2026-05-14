import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import sys

# Importowanie modułów logicznych
from training_data import TrainingData
from vision_processor import VisionProcessor, mp_pose
from audio_engine import AudioEngine


class PersonalTrainerApp:
    def __init__(self, window):
        if sys.version_info.major != 3 or sys.version_info.minor != 11:
            print("BŁĄD: Projekt wymaga Pythona 3.11.x")
            sys.exit(1)

        self.window = window
        self.window.title("AI SUMO TRAINER")
        self.window.geometry("1280x800")

        self.is_dark = True
        self.stat_cards = []  # Przechowuje referencje do kart statystyk dla zmiany motywu

        self.data_manager = TrainingData()
        self.vision = VisionProcessor()
        self.audio = AudioEngine(self.window, self.start_training, self.stop_training)

        self.counter = 0
        self.stage = None
        self.is_running = False
        self.calibration_done = False
        self.target_depth = 90.0

        self.cap = None
        self.pose = None

        self._setup_styles()
        self.window.configure(bg=self.colors["bg"])
        self._create_layout()

    def _setup_styles(self):
        """Definicja palet kolorów dla trybu jasnego i ciemnego."""
        self.palettes = {
            "dark": {
                "bg": "#121212", "card": "#1E1E1E", "card_inner": "#252526",
                "accent": "#00E676", "danger": "#FF5252", "info": "#2979FF",
                "text": "#FFFFFF", "text_dim": "#B0B0B0"
            },
            "light": {
                "bg": "#F5F7FA", "card": "#FFFFFF", "card_inner": "#E4E7EB",
                "accent": "#00C853", "danger": "#D50000", "info": "#2962FF",
                "text": "#212121", "text_dim": "#5F6368"
            }
        }
        self.colors = self.palettes["dark"]

        self.fonts = {
            "title": ("Segoe UI", 24, "bold"),
            "header": ("Segoe UI", 14, "bold"),
            "stat_val": ("Segoe UI", 32, "bold"),
            "stat_label": ("Segoe UI", 10),
            "btn": ("Segoe UI", 11, "bold")
        }

    def toggle_theme(self):
        """Płynnie przełącza kolory całego interfejsu."""
        self.is_dark = not self.is_dark
        self.colors = self.palettes["dark"] if self.is_dark else self.palettes["light"]

        # Aktualizacja głównych kontenerów
        self.window.configure(bg=self.colors["bg"])
        self.sidebar.configure(bg=self.colors["card"])
        self.main_content.configure(bg=self.colors["bg"])
        self.video_container.configure(highlightbackground=self.colors["card_inner"])
        self.status_bar.configure(bg=self.colors["card"])

        # Aktualizacja tekstów
        self.title_label.configure(bg=self.colors["card"])
        self.subtitle_label.configure(bg=self.colors["card"], fg=self.colors["text_dim"])
        self.student_label.configure(bg=self.colors["card"])
        self.status_text.configure(bg=self.colors["card"], fg=self.colors["text_dim"])
        self.video_label.configure(fg=self.colors["card_inner"])

        # Aktualizacja kart
        for card, title, val in self.stat_cards:
            card.configure(bg=self.colors["card_inner"])
            title.configure(bg=self.colors["card_inner"], fg=self.colors["text_dim"])
            val.configure(bg=self.colors["card_inner"], fg=self.colors["text"])

        # Aktualizacja przycisku motywu
        self.btn_theme.configure(
            text="☀️ TRYB JASNY" if self.is_dark else "🌙 TRYB CIEMNY",
            bg="#333333" if self.is_dark else "#E0E0E0",
            fg="#FFFFFF" if self.is_dark else "#000000"
        )

    def _create_layout(self):
        self.sidebar = tk.Frame(self.window, bg=self.colors["card"], width=300)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.title_label = tk.Label(self.sidebar, text="SUMO AI", font=self.fonts["title"],
                                    bg=self.colors["card"], fg=self.colors["accent"])
        self.title_label.pack(pady=(40, 5))

        self.subtitle_label = tk.Label(self.sidebar, text="PERSONAL TRAINER v2.0", font=("Segoe UI", 8),
                                       bg=self.colors["card"], fg=self.colors["text_dim"])
        self.subtitle_label.pack()

        tk.Frame(self.sidebar, bg="#888888", height=1).pack(fill=tk.X, padx=30, pady=20)

        self._create_stat_card(self.sidebar, "POWTÓRZENIA", "0", "counter_label")
        self._create_stat_card(self.sidebar, "CEL (KĄT)", "--", "target_label")

        self.btn_start = self._create_nav_button("ROZPOCZNIJ SESJĘ", self.colors["accent"], self.start_training)
        self.btn_stop = self._create_nav_button("ZAKOŃCZ I ZAPISZ", self.colors["danger"], self.stop_training)
        self._create_nav_button("HISTORIA TRENINGÓW", self.colors["info"], self.data_manager.show_stats)

        # Przycisk zmiany motywu
        self.btn_theme = tk.Button(self.sidebar, text="☀️ TRYB JASNY", command=self.toggle_theme,
                                   bg="#333333", fg="white", font=("Segoe UI", 10, "bold"), relief="flat",
                                   cursor="hand2")
        self.btn_theme.pack(pady=20, padx=30, fill=tk.X)

        self.student_label = tk.Label(self.sidebar, text="Projekt Zespołowy - KCK", font=("Segoe UI", 9),
                                      bg=self.colors["card"], fg="#888888")
        self.student_label.pack(side=tk.BOTTOM, pady=20)

        self.main_content = tk.Frame(self.window, bg=self.colors["bg"])
        self.main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=40, pady=40)

        self.video_container = tk.Frame(self.main_content, bg="#000000", bd=2, highlightthickness=2,
                                        highlightbackground=self.colors["card_inner"])
        self.video_container.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.video_container, bg="#000000", text="KAMERA GOTOWA",
                                    font=self.fonts["header"], fg=self.colors["card_inner"])
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.status_bar = tk.Frame(self.main_content, bg=self.colors["card"], height=40)
        self.status_bar.pack(fill=tk.X, pady=(20, 0))
        self.status_text = tk.Label(self.status_bar, text="OCZEKIWANIE NA KOMENDĘ GŁOSOWĄ...",
                                    font=("Segoe UI", 10), bg=self.colors["card"], fg=self.colors["text_dim"])
        self.status_text.pack(side=tk.LEFT, padx=20)

    def _create_stat_card(self, parent, label, value, attr_name):
        card = tk.Frame(parent, bg=self.colors["card_inner"], padx=15, pady=15)
        card.pack(fill=tk.X, padx=30, pady=10)

        l_title = tk.Label(card, text=label, font=self.fonts["stat_label"], bg=self.colors["card_inner"],
                           fg=self.colors["text_dim"])
        l_title.pack(anchor="w")

        l_val = tk.Label(card, text=value, font=self.fonts["stat_val"], bg=self.colors["card_inner"],
                         fg=self.colors["text"])
        l_val.pack(anchor="w")

        setattr(self, attr_name, l_val)
        self.stat_cards.append((card, l_title, l_val))

    def _create_nav_button(self, text, color, command):
        btn = tk.Button(self.sidebar, text=text, command=command, bg=color, fg="white",
                        font=self.fonts["btn"], relief="flat", cursor="hand2", bd=0, width=22, height=2)
        btn.pack(pady=8, padx=30)
        return btn

    def start_training(self):
        if self.is_running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Błąd", "Nie znaleziono kamery.")
            return

        self.is_running, self.counter, self.calibration_done = True, 0, False
        self.counter_label.config(text="0")
        self.target_label.config(text="CALIB")

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.status_text.config(text="SESJA AKTYWNA: TRWA KALIBRACJA", fg=self.colors["accent"])
        self.audio.speak("Rozpoczynamy. Wykonaj trzy powtórzenia kalibracyjne.")
        self.process_video()

    def stop_training(self):
        if not self.is_running: return
        self.is_running = False
        self.status_text.config(text="ZAKOŃCZONO TRENING", fg=self.colors["danger"])

        if self.cap: self.cap.release()
        if self.pose: self.pose.close()

        self.data_manager.save_to_csv(self.counter)
        self.audio.speak(self.data_manager.get_progression_message(self.counter))
        messagebox.showinfo("Sesja zapisana", f"Twój wynik: {self.counter}")

    def process_video(self):
        if not self.is_running or not self.cap: return
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

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.window.after(10, self.process_video)

    def _handle_pose_logic(self, image, lm, results):
        if not self.vision.check_visibility(lm):
            cv2.putText(image, "SKORYGUJ POZYCJE (POZA KADREM)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        2)
            return

        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y,
               lm[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                 lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]

        raw_angle = self.vision.calculate_angle_3d(hip, knee, ankle)
        angle = self.vision.get_smoothed_angle(raw_angle)

        dist_ankle = abs(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x - lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
        dist_hip = abs(lm[mp_pose.PoseLandmark.LEFT_HIP.value].x - lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x)
        is_sumo = dist_ankle > dist_hip * 1.5

        color = (0, 255, 0)

        if not self.calibration_done:
            color = (0, 165, 255)
            if angle < self.current_min_angle: self.current_min_angle = angle
            if angle > 160: self.stage = "gora"
            if angle < 100 and self.stage == "gora" and is_sumo: self.stage = "dol"
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
                    self.audio.speak(f"Kalibracja zakonczona.")
        else:
            self.target_label.config(text=f"{int(self.target_depth)}°")
            if not is_sumo:
                color = (0, 0, 255)
                cv2.putText(image, "SZERZEJ STOPY!", (10, 140), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            if angle > 160: self.stage = "gora"
            if angle < self.target_depth and self.stage == "gora":
                if is_sumo:
                    self.stage = "dol"
                    self.counter += 1
                    color = (0, 255, 0)
                    self.audio.speak(str(self.counter))
                else:
                    self.stage = "blad"
                    self.audio.speak("Szerzej stopy")
            self.counter_label.config(text=str(self.counter))

        self.vision.draw_protractor(image, hip, knee, ankle, angle, color)
        self.vision.draw_landmarks(image, results.pose_landmarks)


if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalTrainerApp(root)
    root.mainloop()