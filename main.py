import customtkinter as ctk
from tkinter import messagebox
import cv2
import numpy as np
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

        self.view = TrainerGuiView(self.window, self.start_session, self.show_summary, self.log_set,
                                   self.discard_session, self.show_tutorial)

        self.state = STATE_DASHBOARD
        self.active_session = None
        self.current_set_id = 1
        self.current_reps = 0
        self.pose_stage = None

        # Zmienne wizyjne
        self.is_dual_cam = False
        self.cap_front = None
        self.cap_side = None
        self.pose_front = None
        self.pose_side = None

        self.view.show_dashboard(self.data_manager.get_dashboard_summary())

    def show_tutorial(self):
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

        cam_mode = self.view.get_cam_mode()
        self.is_dual_cam = (cam_mode == "2 Kamery (Front + Profil)")

        # Inicjalizacja kamery frontowej (domyślnej)
        self.cap_front = cv2.VideoCapture(0)
        self.pose_front = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        if not self.cap_front.isOpened():
            messagebox.showerror("Błąd", "Nie udało się otworzyć głównej kamery.")
            self.state = STATE_DASHBOARD
            return

        # Inicjalizacja drugiej kamery
        if self.is_dual_cam:
            url_or_index = self.view.get_cam_url()
            self.cap_side = cv2.VideoCapture(url_or_index)
            self.pose_side = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            if not self.cap_side.isOpened():
                messagebox.showwarning("Uwaga", "Nie udało się otworzyć kamery profilowej. Kontynuacja z 1 kamerą.")
                self.is_dual_cam = False

        self.view.show_session_running()
        self.view.update_status("Kamery aktywne. Wykonuj przysiady!")
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

        if self.cap_front: self.cap_front.release()
        if self.cap_side: self.cap_side.release()
        if self.pose_front: self.pose_front.close()
        if self.pose_side: self.pose_side.close()

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
        if self.state != STATE_SESSION_RUNNING or not self.cap_front: return

        ret_f, frame_f = self.cap_front.read()
        if not ret_f:
            self.window.after(10, self.process_video)
            return
        frame_f = cv2.flip(frame_f, 1)

        frame_s = None
        if self.is_dual_cam and self.cap_side:
            ret_s, frame_s = self.cap_side.read()
            # Brak flipa dla bocznej, by profil był naturalny

        if self.is_dual_cam and frame_s is not None:
            self._analyze_dual_pose(frame_f, frame_s)
        else:
            self._analyze_single_pose(frame_f)

        if self.state == STATE_SESSION_RUNNING:
            self.window.after(10, self.process_video)

    def _analyze_single_pose(self, frame_f):
        image_rgb = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
        results = self.pose_front.process(image_rgb)
        image = frame_f.copy()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            self._run_biomechanics(image, lm, results, is_dual=False)

        cv2.putText(image, "TRYB: 1 KAMERA", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.view.update_video(pil_img)

    def _analyze_dual_pose(self, frame_f, frame_s):
        # Front
        rgb_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
        res_f = self.pose_front.process(rgb_f)
        img_f = frame_f.copy()

        # Profil (Bok)
        rgb_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
        res_s = self.pose_side.process(rgb_s)
        img_s = frame_s.copy()

        if res_f.pose_landmarks and res_s.pose_landmarks:
            lm_f = res_f.pose_landmarks.landmark
            lm_s = res_s.pose_landmarks.landmark
            self._run_biomechanics_dual(img_f, lm_f, res_f, img_s, lm_s, res_s)

        # Dopasowanie rozmiarów do Split-Screen
        h, w, _ = img_f.shape
        target_h, target_w = 480, 410  # Połowa z 820 szerokości z gui_view

        img_f_resized = cv2.resize(img_f, (target_w, target_h))
        img_s_resized = cv2.resize(img_s, (target_w, target_h))

        cv2.putText(img_f_resized, "FRONT", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(img_s_resized, "PROFIL", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        combined_frame = np.hstack((img_f_resized, img_s_resized))
        pil_img = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
        self.view.update_video(pil_img)

    def _run_biomechanics(self, image, lm, results, is_dual=False):
        if not self.vision.check_visibility(lm):
            cv2.putText(image, "POZA KADREM!", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            return

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

        raw_angle = self.vision.calculate_angle_3d(left_hip, left_knee, left_ankle)
        knee_angle = self.vision.get_smoothed_angle(raw_angle)
        hip_extension_angle = self.vision.calculate_angle_3d(left_shoulder, left_hip, left_knee)

        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        ankle_width = abs(left_ankle[0] - right_ankle[0])
        knee_width = abs(left_knee[0] - right_knee[0])

        mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        torso_angle = self.vision.calculate_angle_2d_vertical(mid_shoulder, mid_hip)

        is_wide_stance = ankle_width > 1.3 * shoulder_width
        is_knee_valgus = knee_width < 0.7 * ankle_width
        is_torso_too_bent = torso_angle > 45.0
        is_deep_enough = left_hip[1] >= (left_knee[1] - 0.05)

        color = (0, 255, 0)
        feedback_msg = ""

        if not is_wide_stance:
            color = (0, 0, 255)
            feedback_msg = "SZERZEJ STOPY!"
        elif is_torso_too_bent:
            color = (0, 165, 255)
            feedback_msg = "WYPROSTUJ PLECY!"
        elif is_knee_valgus and self.pose_stage == "dol":
            color = (0, 0, 255)
            feedback_msg = "KOLANA NA ZEWNATRZ!"

        if knee_angle > 160 and hip_extension_angle > 160:
            self.pose_stage = "gora"

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

        self.vision.draw_protractor(image, left_hip, left_knee, left_ankle, knee_angle, color)
        self.vision.draw_landmarks(image, results.pose_landmarks)

        if feedback_msg:
            cv2.putText(image, feedback_msg, (10, 140), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.putText(image, f"Seria {self.current_set_id}: {self.current_reps} powt.", (10, 70), cv2.FONT_HERSHEY_DUPLEX,
                    1.2, (0, 255, 255), 2)

    def _run_biomechanics_dual(self, img_f, lm_f, res_f, img_s, lm_s, res_s):
        # Kamera Front: Tylko oś X (szerokości)
        left_sh_f = lm_f[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_sh_f = lm_f[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_knee_f = lm_f[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee_f = lm_f[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle_f = lm_f[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle_f = lm_f[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        shoulder_width = abs(left_sh_f.x - right_sh_f.x)
        ankle_width = abs(left_ankle_f.x - right_ankle_f.x)
        knee_width = abs(left_knee_f.x - right_knee_f.x)

        is_wide_stance = ankle_width > 1.3 * shoulder_width
        is_knee_valgus = knee_width < 0.7 * ankle_width

        # Kamera Bok: Oś Y, Z, głębokość, kąty (zakładamy kamerę na lewy bok ćwiczącego)
        left_shoulder = [lm_s[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         lm_s[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         lm_s[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        left_hip = [lm_s[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm_s[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    lm_s[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        left_knee = [lm_s[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm_s[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                     lm_s[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        left_ankle = [lm_s[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm_s[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                      lm_s[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]

        raw_angle = self.vision.calculate_angle_3d(left_hip, left_knee, left_ankle)
        knee_angle = self.vision.get_smoothed_angle(raw_angle)
        hip_extension_angle = self.vision.calculate_angle_3d(left_shoulder, left_hip, left_knee)

        torso_angle = self.vision.calculate_angle_2d_vertical(left_shoulder, left_hip)

        is_torso_too_bent = torso_angle > 45.0
        is_deep_enough = left_hip[1] >= (left_knee[1] - 0.05)

        color = (0, 255, 0)
        feedback_msg = ""

        if not is_wide_stance:
            color = (0, 0, 255)
            feedback_msg = "SZERZEJ STOPY!"
        elif is_torso_too_bent:
            color = (0, 165, 255)
            feedback_msg = "WYPROSTUJ PLECY!"
        elif is_knee_valgus and self.pose_stage == "dol":
            color = (0, 0, 255)
            feedback_msg = "KOLANA NA ZEWNATRZ!"

        if knee_angle > 160 and hip_extension_angle > 160:
            self.pose_stage = "gora"

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

        self.vision.draw_landmarks(img_f, res_f.pose_landmarks)
        self.vision.draw_protractor(img_s, left_hip, left_knee, left_ankle, knee_angle, color)
        self.vision.draw_landmarks(img_s, res_s.pose_landmarks)

        if feedback_msg:
            cv2.putText(img_f, feedback_msg, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img_s, feedback_msg, (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        cv2.putText(img_s, f"Seria {self.current_set_id}: {self.current_reps}", (10, 130), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                    (0, 255, 255), 2)


if __name__ == "__main__":
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        print("BŁĄD: Użyj Pythona 3.11.x")
        sys.exit(1)

    root = ctk.CTk()
    app = PersonalTrainerApp(root)
    root.mainloop()