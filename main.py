import customtkinter as ctk
from tkinter import messagebox
import cv2
from PIL import Image
import sys

from training_data import TrainingHistoryGuided
from gui_view import TrainerGuiView
from vision_processor import VisionProcessor, mp_pose
from audio_engine import AudioEngine

STATE_DASHBOARD = "DASHBOARD"
STATE_SESSION_RUNNING = "SESSION_RUNNING"
STATE_SESSION_SUMMARY = "SESSION_SUMMARY"

class PersonalTrainerApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI SUMO TRAINER")
        self.window.geometry("1280x800")

        self.data_manager = TrainingHistoryGuided()
        self.vision = VisionProcessor()
        self.audio = AudioEngine(self.window, self.start_session, self.finish_and_save)

        # Przekazujemy nową metodę do obsługi przycisku tutorial
        self.view = TrainerGuiView(self.window, self.start_session, self.show_summary, self.log_set,
                                   self.discard_session, self.show_tutorial)

        self.state = STATE_DASHBOARD
        self.active_session = None
        self.current_set_id = 1
        self.current_reps = 0
        self.pose_stage = None

        self.cap = None
        self.pose = None

        self.view.show_dashboard(self.data_manager.get_dashboard_summary())

    def show_tutorial(self):
        """Metoda wywoływana po kliknięciu przycisku TUTORIAL. Pozostawiona pusta zgodnie z zaleceniem."""
        pass

    def start_session(self):
        if self.state != STATE_DASHBOARD: return

        from datetime import datetime
        self.active_session = {
            'session_name': f"Trening {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'sets_list': [],
            'total_reps': 0
        }
        self.current_set_id = 1
        self.current_reps = 0
        self.pose_stage = None
        self.state = STATE_SESSION_RUNNING

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Błąd", "Nie udało się otworzyć kamery.")
            self.state = STATE_DASHBOARD
            return

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.view.show_session_running()
        self.view.update_status("Kamera aktywna. Wykonuj przysiady!")
        self.audio.speak("Rozpoczynamy nową sesję. Zaczynam analizę.")
        self.process_video()

    def log_set(self):
        if self.state != STATE_SESSION_RUNNING: return

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

    def show_summary(self):
        if self.state != STATE_SESSION_RUNNING: return

        self.state = STATE_SESSION_SUMMARY
        if self.cap: self.cap.release()
        if self.pose: self.pose.close()

        self.view.show_summary(self.active_session)
        self.audio.speak("Sesja zakończona. Sprawdź podsumowanie i zapisz.")

    def finish_and_save(self):
        if self.state != STATE_SESSION_SUMMARY: return

        success = self.data_manager.save_complete_session(self.active_session)
        if success:
            msg = self.data_manager.get_structured_session_string(self.active_session['total_reps'],
                                                                  len(self.active_session['sets_list']))
            self.audio.speak(msg)
            messagebox.showinfo("Sukces", "Zapisano Pomyślnie! Twoja historia została zaktualizowana.")
        else:
            messagebox.showerror("Błąd", "Nie udało się zapisać pliku CSV.")

        self.state = STATE_DASHBOARD
        self.active_session = None
        self.view.show_dashboard(self.data_manager.get_dashboard_summary())

    def discard_session(self):
        if self.state != STATE_SESSION_SUMMARY: return
        self.state = STATE_DASHBOARD
        self.active_session = None
        self.view.show_dashboard(self.data_manager.get_dashboard_summary())
        self.audio.speak("Sesja odrzucona.")

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
            self._analyze_pose(image, lm, results)

        image_rgb_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb_show)
        self.view.update_video(pil_img)

        if self.state == STATE_SESSION_RUNNING:
            self.window.after(10, self.process_video)

    def _analyze_pose(self, image, lm, results):
        if not self.vision.check_visibility(lm):
            cv2.putText(image, "POZA KADREM!", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            return

        # 1. Pobieranie punktów strukturalnych 3D
        left_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

        left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        right_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                     lm[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

        left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                     lm[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        right_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                      lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]

        left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                      lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        right_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                       lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

        # 2. Zaawansowane metryki biomechaniczne
        raw_angle = self.vision.calculate_angle_3d(left_hip, left_knee, left_ankle)
        knee_angle = self.vision.get_smoothed_angle(raw_angle)

        hip_extension_angle = self.vision.calculate_angle_3d(left_shoulder, left_hip, left_knee)

        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        ankle_width = abs(left_ankle[0] - right_ankle[0])
        knee_width = abs(left_knee[0] - right_knee[0])

        # Liczenie kąta nachylenia pleców na podstawie rzutowania środków geometrycznych
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        torso_angle = self.vision.calculate_angle_2d_vertical(mid_shoulder, mid_hip)

        # 3. Sprawdzanie progów rygorystycznych dla wersji Sumo
        is_wide_stance = ankle_width > 1.3 * shoulder_width
        is_knee_valgus = knee_width < 0.7 * ankle_width
        is_torso_too_bent = torso_angle > 45.0
        is_deep_enough = left_hip[1] >= (left_knee[1] - 0.05)

        color = (0, 255, 0)
        feedback_msg = ""

        # Przypisanie alertów wizualnych
        if not is_wide_stance:
            color = (0, 0, 255)
            feedback_msg = "SZERZEJ STOPY!"
        elif is_torso_too_bent:
            color = (0, 165, 255)
            feedback_msg = "WYPROSTUJ PLECY!"
        elif is_knee_valgus and self.pose_stage == "dol":
            color = (0, 0, 255)
            feedback_msg = "KOLANA NA ZEWNATRZ!"

        # 4. Maszyna stanów licznika
        # Warunek górny (pełny wyprosty kolan i bioder)
        if knee_angle > 160 and hip_extension_angle > 160:
            self.pose_stage = "gora"

        # Warunek dolny (głębokość miednicy)
        if knee_angle < 100 and self.pose_stage == "gora":
            if is_deep_enough:
                if is_wide_stance and not is_knee_valgus and not is_torso_too_bent:
                    self.pose_stage = "dol"
                    self.current_reps += 1
                    self.audio.speak(str(self.current_reps))
                    feedback_msg = "DOBRZE!"
                else:
                    self.pose_stage = "blad"
                    if not is_wide_stance:
                        self.audio.speak("Szerzej stopy")
                    elif is_knee_valgus:
                        self.audio.speak("Kolana na zewnatrz")
                    elif is_torso_too_bent:
                        self.audio.speak("Wyprostuj plecy")

        # Rysowanie na ekranie
        self.vision.draw_protractor(image, left_hip, left_knee, left_ankle, knee_angle, color)
        self.vision.draw_landmarks(image, results.pose_landmarks)

        if feedback_msg:
            cv2.putText(image, feedback_msg, (10, 140), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        cv2.putText(image, f"Seria {self.current_set_id}: {self.current_reps} powt.", (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1.2, (0, 255, 255), 2)


if __name__ == "__main__":
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        print("BŁĄD: Użyj Pythona 3.11.x")
        sys.exit(1)

    root = ctk.CTk()
    app = PersonalTrainerApp(root)
    root.mainloop()