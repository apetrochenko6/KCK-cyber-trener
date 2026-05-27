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

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
            cv2.putText(
                image,
                "SKORYGUJ POZYCJE (POZA KADREM)",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            return

        hip = [
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].z
        ]

        knee = [
            lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z
        ]

        ankle = [
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].z
        ]

        raw_angle = self.vision.calculate_angle_3d(hip, knee, ankle)
        angle = self.vision.get_smoothed_angle(raw_angle)

        dist_ankle = abs(
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x -
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
        )

        dist_hip = abs(
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].x -
            lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x
        )

        dist_knee = abs(
            lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x -
            lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
        )

        feet_wide = dist_ankle > dist_hip * 1.2
        knees_out = dist_knee > (dist_ankle * 0.83)
        print(f"Knees Out: {knees_out}, Feet Wide: {feet_wide}")
        angle = self.vision.get_smoothed_angle(raw_angle)
        print(f"Angle: {int(angle)} | Stage: {self.stage}")
        print(f"Ankle width: {dist_ankle:.2f} | Knee width: {dist_knee:.2f}")
        is_sumo = feet_wide and knees_out
        color = (0, 255, 0)
        if not self.calibration_done:
            color = (0, 165, 255)
            if angle < self.current_min_angle:
                self.current_min_angle = angle
            if self.stage == "dol" and angle > 115:
                if is_sumo:
                    self.stage = "gora"
                    self.calibration_angles.append(self.current_min_angle)
                    self.calibration_count += 1
                    self.current_min_angle = 180.0
                    self.audio.speak(str(self.calibration_count))
                    if self.calibration_count >= self.calibration_reps:
                        self.target_depth = (sum(self.calibration_angles) / len(self.calibration_angles)) + 15
                        self.calibration_done = True
                        self.stage = None
                        self.audio.speak("Kalibracja zakonczona.")
                else:
                    self.stage = "blad"
            elif angle > 115:
                if is_sumo:
                    self.stage = "gora"
                else:
                    self.stage = "blad"

            if self.stage == "gora" and angle < 85 and is_sumo:
                self.stage = "dol"

        else:
            self.view.update_target(f"{int(self.target_depth)}°")

            if not is_sumo:
                color = (0, 0, 255)
                cv2.putText(
                    image,
                    "ROZSUN KOLANA LUB STOPY!",
                    (10, 140),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            if angle > 115:
                if is_sumo:
                    self.stage = "gora"
                else:
                    self.stage = "blad"

            if angle < self.target_depth and self.stage == "gora":
                if is_sumo:
                    self.stage = "dol"
                    self.counter += 1
                    color = (0, 255, 0)
                    self.audio.speak(str(self.counter))
                else:
                    self.stage = "blad"
                    self.audio.speak("Zla postawa")

            self.view.update_counter(str(self.counter))

        self.vision.draw_protractor(image, hip, knee, ankle, angle, color)
        self.vision.draw_landmarks(image, results.pose_landmarks)


if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalTrainerApp(root)
    root.mainloop()