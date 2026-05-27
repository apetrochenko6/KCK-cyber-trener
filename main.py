import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import sys

# Importowanie modułów logicznych
from training_data import TrainingData
from vision_processor import VisionProcessor, mp_pose
from audio_engine import AudioEngine
from gui_view import TrainerGuiView
STATE_DASHBOARD = "DASHBOARD"
STATE_SESSION_RUNNING = "SESSION_RUNNING"
STATE_SESSION_SUMMARY = "SESSION_SUMMARY"
class PersonalTrainerApp:
    def __init__(self, window):
        if sys.version_info.major != 3 or sys.version_info.minor != 11:
            print("BŁĄD: Projekt wymaga Pythona 3.11.x")
            sys.exit(1)
        self.state = STATE_DASHBOARD
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
        self.view = TrainerGuiView(
            self.window,
            start_cmd=self.start_training,
            stop_cmd=self.stop_training,
            log_cmd=self.log_set,
            discard_cmd=self.discard_session,
            tutorial_cmd=self.show_tutorial
        )
        self.current_set_id = 1
        self.current_reps = 0
        # Kalibracja przysiadu
        self.calibration_reps = 3
        self.calibration_count = 0
        self.calibration_angles = []
        self.current_min_angle = 180.0

        self.cap = None
        self.pose = None

        self.view.show_dashboard(self.data_manager.get_dashboard_summary())

    def log_set(self):
        if self.state != STATE_SESSION_RUNNING: return
        if not self.calibration_done:
            messagebox.showwarning("Uwaga", "Najpierw dokończ kalibrację!")
            return

        target = self.view.get_target_reps()

        set_data = {
            'set_id': self.current_set_id,
            'target_reps': target,
            'logged_reps': self.current_reps,
            'completed': self.current_reps >= target
        }

        self.active_session['sets_list'].append(set_data)
        self.active_session['total_reps'] += self.current_reps

        self.current_set_id += 1
        self.current_reps = 0

        self.view.update_live_sets(self.active_session['sets_list'])
        self.view.update_status(f"Zalogowano serię. Zaczynamy serię {self.current_set_id}.")
        self.audio.speak("Seria zalogowana.")

    def save_session(self):
        success = self.data_manager.save_complete_session(self.active_session)
        if success:
            msg = self.data_manager.get_structured_session_string(self.active_session['total_reps'],
                                                                  len(self.active_session['sets_list']))
            self.audio.speak(msg)
            messagebox.showinfo("Sesja zapisana",
                                f"Zapisano pomyślnie!\nWszystkie powtórzenia: {self.active_session['total_reps']}")
        else:
            messagebox.showerror("Błąd", "Nie udało się zapisać pliku CSV.")
        self.state = STATE_DASHBOARD
        self.active_session = None
        self.view.show_dashboard(self.data_manager.get_dashboard_summary())
    def start_training(self):
        if self.state != STATE_DASHBOARD: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Błąd", "Nie znaleziono kamery.")
            return

        self.state = STATE_SESSION_RUNNING
        self.is_running = True
        self.active_session = {
            'sets_list': [],
            'total_reps': 0
        }

        self.counter = 0
        self.stage = None
        self.calibration_done = False

        # Reset kalibracji
        self.calibration_count = 0
        self.calibration_angles = []
        self.current_min_angle = 180.0

        self.view.update_counter("0")
        self.view.update_target("CALIB")

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
        self.view.show_session_running()
        self.view.update_status("Kamera aktywna. Wykonuj przysiady!")
        self.audio.speak("Rozpoczynamy nową sesję. Zaczynam analizę.")
        self.process_video()
    def show_tutorial(self):
        pass
    def discard_session(self):
        if self.state != STATE_SESSION_SUMMARY: return
        self.state = STATE_DASHBOARD
        self.active_session = None
        self.view.show_dashboard(self.data_manager.get_dashboard_summary())
        self.audio.speak("Sesja odrzucona.")
    def stop_training(self):
        if not self.is_running: return
        self.is_running = False
        self.state = STATE_SESSION_SUMMARY
        self.view.update_status("ZAKOŃCZONO TRENING")
        if self.cap: self.cap.release()
        if self.pose: self.pose.close()

        self.view.show_summary(self.active_session)
        messagebox.showinfo("Sesja zapisana", f"Twój wynik: {self.counter}")
        self.save_session()

    def process_video(self):
        if self.state != STATE_SESSION_RUNNING or not self.cap: return
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
        image_rgb_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb_show)
        self.view.update_video(pil_img)

        if self.state == STATE_SESSION_RUNNING:
            self.window.after(10, self.process_video)

    def _handle_pose_logic(self, image, lm, results):
        if not self.vision.check_visibility(lm):
            cv2.putText(image, "POZA KADREM", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.stage = 'blad'
            return

        # =========================
        # 1. LANDMARKS
        # =========================
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        dist_shoulder = self.vision.get_3d_dist(l_shoulder, r_shoulder)
        hip = [l_hip.x, l_hip.y, l_hip.z]
        knee = [l_knee.x, l_knee.y, l_knee.z]
        ankle = [l_ankle.x, l_ankle.y, l_ankle.z]

        raw_angle = self.vision.calculate_angle_3d(hip, knee, ankle)
        angle = self.vision.get_smoothed_angle(raw_angle)

        if not hasattr(self, "prev_angle"):
            self.prev_angle = angle
        velocity = angle - self.prev_angle
        self.prev_angle = angle

        dist_ankle = self.vision.get_3d_dist(l_ankle, r_ankle)
        dist_knee = self.vision.get_3d_dist(l_knee, r_knee)
        dist_hip = self.vision.get_3d_dist(l_hip, r_hip)

        knee_ratio = dist_knee / (dist_shoulder * 1.2 + 0.0001)

        feet_ratio = dist_ankle / (dist_shoulder * 1.5 + 0.0001)

        knees_ok = knee_ratio > 1.0
        feet_wide = feet_ratio > 1.0
        is_sumo = feet_wide and knees_ok

        if not is_sumo:
            cv2.putText(image, "BLAD POSTAWY!", (10, 140), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        target_depth = 95

        if self.stage == "gora" and angle < target_depth:
            if is_sumo:
                self.stage = "dol"
            else:
                self.stage = "blad"
                self.audio.speak("Zla postawa")
        if angle > 110:
            if is_sumo:
                if self.stage == "dol":
                    self.stage = "gora"
                    self.counter += 1
                    self.current_reps += 1
                    self.audio.speak(str(self.counter))
                if self.stage == "blad":
                    self.stage = "gora"
            else:
                self.stage = "blad"
                cv2.putText(image, "POPRAW POSTAWE!", (10, 140), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)


        # =========================
        # LOGS
        # =========================
        print("-" * 40)
        print(f"Angle: {int(angle)}° | Velocity: {velocity:.2f} | Stage: {self.stage}")
        print("--- Widths (3D) ---")
        print(f"Ankle: {dist_ankle:.3f} | Knee: {dist_knee:.3f} | Hip: {dist_hip:.3f}")
        print("--- Ratios ---")
        print(f"Feet/Hip Ratio: {feet_ratio:.3f} (Треба > 1.5) -> {feet_wide}")
        print(f"Knee/Hip Ratio: {knee_ratio:.3f} (Треба > 1.2) -> {knees_ok}")
        print("--- Final ---")
        print(f"IS SUMO: {is_sumo}")

        # =========================
        # 6. VISUAL
        # =========================
        color = (0, 255, 0) if is_sumo else (0, 0, 255)

        self.view.update_counter(str(self.counter))
        self.vision.draw_protractor(image, hip, knee, ankle, angle, color)
        self.vision.draw_landmarks(image, results.pose_landmarks)


if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalTrainerApp(root)
    root.mainloop()