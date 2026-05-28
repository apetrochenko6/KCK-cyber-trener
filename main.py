from tkinter import messagebox
import cv2
from PIL import Image
import sys
import webbrowser
import customtkinter as ctk
from training_data import TrainingData
from vision_processor import VisionProcessor, mp_pose
from audio_engine import AudioEngine
from gui_view import TrainerGuiView

STATE_DASHBOARD = "DASHBOARD"
STATE_SESSION_RUNNING = "SESSION_RUNNING"
STATE_SESSION_SUMMARY = "SESSION_SUMMARY"


class PersonalTrainerApp:
    DOWN_CALIBRATION_LIMIT = 120
    UP_THRESHOLD = 150
    MIN_TARGET_DEPTH = 75
    MAX_TARGET_DEPTH = 105
    DEPTH_TOLERANCE = 7

    def __init__(self, window):
        if sys.version_info.major != 3 or sys.version_info.minor != 11:
            print("BŁĄD: Projekt wymaga Pythona 3.11.x")
            sys.exit(1)
        self.state = STATE_DASHBOARD
        self.window = window
        self.window.title("CyberTrainer")
        self.window.geometry("1280x800")
        self.data_manager = TrainingData()
        self.vision = VisionProcessor()
        self.counter = 0
        self.stage = "gora"
        self.is_running = False
        self.calibration_done = False
        self.target_depth = 95.0
        self.current_set_id = 1
        self.current_reps = 0
        self.calibration_reps = 3
        self.calibration_count = 0
        self.calibration_angles = []
        self.current_min_angle = 180.0
        self.active_session = None
        self.cap = None
        self.pose = None
        self.view = TrainerGuiView(
            self.window,
            start_cmd=self.start_training,
            stop_cmd=self.stop_training,
            save_cmd=self.save_session,
            log_cmd=self.log_set,
            discard_cmd=self.discard_session,
            tutorial_cmd=self.show_tutorial
        )
        self.audio = AudioEngine(self.window, self.start_training, self.stop_training)
        self.view.show_dashboard(self.data_manager.get_dashboard_summary())

    def log_set(self):
        if self.state != STATE_SESSION_RUNNING:
            return
        if not self.calibration_done:
            messagebox.showwarning("Uwaga", "Najpierw dokończ kalibrację.")
            return
        if not self._finalize_current_set():
            messagebox.showwarning("Uwaga", "Nie ma powtórzeń do zapisania w tej serii.")
            return
        self.view.update_status(f"Zalogowano serię. Zaczynamy serię {self.current_set_id}.")
        self.audio.speak("Seria zalogowana")

    def _finalize_current_set(self):
        if self.current_reps <= 0 or self.active_session is None:
            return False
        target = self.view.get_target_reps()
        set_data = {
            "set_id": self.current_set_id,
            "target_reps": target,
            "logged_reps": self.current_reps,
            "completed": self.current_reps >= target
        }
        self.active_session["sets_list"].append(set_data)
        self.active_session["total_reps"] += self.current_reps
        self.current_set_id += 1
        self.current_reps = 0
        self.view.update_live_sets(self.active_session["sets_list"])
        return True

    def save_session(self):
        if self.state != STATE_SESSION_SUMMARY or self.active_session is None:
            return
        total_reps = self.active_session["total_reps"]
        sets_count = len(self.active_session["sets_list"])
        if total_reps <= 0:
            messagebox.showwarning("Brak danych", "Brak powtórzeń do zapisania.")
            self.discard_session()
            return
        self.data_manager.save_to_csv(total_reps, sets_count)
        msg = self.data_manager.get_progression_message(total_reps)
        self.audio.speak(msg)
        messagebox.showinfo("Sesja zapisana", f"Zapisano pomyślnie.\n{msg}")
        self.state = STATE_DASHBOARD
        self.active_session = None
        self.view.show_dashboard(self.data_manager.get_dashboard_summary())

    def start_training(self):
        if self.state != STATE_DASHBOARD:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Błąd", "Nie znaleziono kamery.")
            return
        self.state = STATE_SESSION_RUNNING
        self.is_running = True
        self.active_session = {"sets_list": [], "total_reps": 0}
        self.counter = 0
        self.stage = "gora"
        self.current_set_id = 1
        self.current_reps = 0
        self.calibration_done = False
        self.calibration_count = 0
        self.calibration_angles = []
        self.current_min_angle = 180.0
        self.target_depth = 95.0
        self.vision.reset_smoothing()
        self.view.update_counter("0")
        self.view.update_target(f"KAL 0/{self.calibration_reps}")
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
        self.view.show_session_running()
        self.view.update_status("Kalibracja: wykonaj 3 spokojne, poprawne przysiady sumo.")
        self.audio.speak("Rozpoczynamy kalibrację. Wykonaj trzy poprawne przysiady sumo.")
        self.process_video()

    def show_tutorial(self):
        tut_win = ctk.CTkToplevel(self.window)
        tut_win.title("Instrukcja - Sumo Squat")
        tut_win.geometry("650x750")
        tut_win.attributes("-topmost", True)
        ctk.CTkLabel(tut_win, text="Jak poprawnie wykonać Sumo Squat", font=("Segoe UI", 20, "bold")).pack(pady=20)
        instr_frame = ctk.CTkFrame(tut_win, fg_color="transparent")
        instr_frame.pack(pady=10, padx=20, fill="x")
        text = (
            "1. Stań szeroko: stopy szerzej niż barki, palce lekko na zewnątrz.\n"
            "2. Trzymaj plecy proste i klatkę piersiową uniesioną.\n"
            "3. Schodź w dół kontrolowanie, a kolana prowadź na zewnątrz."
        )
        ctk.CTkLabel(instr_frame, text=text, font=("Segoe UI", 15), justify="left").pack(anchor="w")
        img_frame = ctk.CTkFrame(tut_win, fg_color="transparent")
        img_frame.pack(pady=20)
        img_start = ctk.CTkImage(light_image=Image.open("assets/tutorial_start_pos.png"), dark_image=Image.open("assets/tutorial_start_pos.png"), size=(250, 250))
        img_squat = ctk.CTkImage(light_image=Image.open("assets/tutorial_end_pos.png"), dark_image=Image.open("assets/tutorial_end_pos.png"), size=(250, 250))
        for img, label in [(img_start, "Pozycja startowa"), (img_squat, "Pozycja dolna")]:
            card = ctk.CTkFrame(img_frame, corner_radius=10)
            card.pack(side="left", padx=15)
            ctk.CTkLabel(card, image=img, text="").pack(pady=5, padx=5)
            ctk.CTkLabel(card, text=label, font=("Segoe UI", 12, "bold")).pack(pady=5)

        def open_video():
            webbrowser.open("https://www.youtube.com/results?search_query=sumo+squat+proper+form")

        ctk.CTkButton(tut_win, text="▶ Obejrzyj wideo tutorial", command=open_video, fg_color="#E64A19", hover_color="#BF360C", height=40).pack(pady=25, padx=50, fill="x")
        ctk.CTkButton(tut_win, text="Zamknij", command=tut_win.destroy, fg_color="transparent", border_width=2, border_color="gray").pack(pady=5)

    def discard_session(self):
        if self.state != STATE_SESSION_SUMMARY:
            return
        self.state = STATE_DASHBOARD
        self.active_session = None
        self.view.show_dashboard(self.data_manager.get_dashboard_summary())
        self.audio.speak("Sesja odrzucona")

    def stop_training(self):
        if not self.is_running:
            return
        self.is_running = False
        self.state = STATE_SESSION_SUMMARY
        self._release_camera()
        if self.calibration_done:
            self._finalize_current_set()
        self.view.update_status("ZAKOŃCZONO TRENING")
        self.view.show_summary(self.active_session)

    def _release_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.pose:
            self.pose.close()
            self.pose = None

    def process_video(self):
        if self.state != STATE_SESSION_RUNNING or not self.cap:
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
        else:
            self.view.update_status("Nie widzę sylwetki. Stań dalej od kamery.")
            cv2.putText(image, "BRAK SYLWETKI", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        image_rgb_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb_show)
        self.view.update_video(pil_img)
        if self.state == STATE_SESSION_RUNNING:
            self.window.after(10, self.process_video)

    def _handle_pose_logic(self, image, lm, results):
        if not self.vision.check_visibility(lm):
            self.stage = "blad"
            self.view.update_status("Część ciała jest poza kadrem. Pokaż biodra, kolana i stopy.")
            cv2.putText(image, "POZA KADREM", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            self.vision.draw_landmarks(image, results.pose_landmarks)
            return
        metrics = self._get_pose_metrics(lm)
        if not self.calibration_done:
            self._handle_calibration(image, metrics)
        else:
            self._handle_training(image, metrics)
        self._draw_feedback(image, metrics, results)

    def _get_pose_metrics(self, lm):
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
        r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = self.vision.landmark_to_point(l_hip)
        right_hip = self.vision.landmark_to_point(r_hip)
        left_knee = self.vision.landmark_to_point(l_knee)
        right_knee = self.vision.landmark_to_point(r_knee)
        left_ankle = self.vision.landmark_to_point(l_ankle)
        right_ankle = self.vision.landmark_to_point(r_ankle)
        left_angle = self.vision.calculate_angle_3d(left_hip, left_knee, left_ankle)
        right_angle = self.vision.calculate_angle_3d(right_hip, right_knee, right_ankle)
        raw_angle = (left_angle + right_angle) / 2
        angle = self.vision.get_smoothed_value("knee_angle", raw_angle, 0.18)
        shoulder_width = abs(l_shoulder.x - r_shoulder.x) + 0.0001
        feet_width = abs(l_ankle.x - r_ankle.x)
        knee_width = abs(l_knee.x - r_knee.x)
        feet_ratio = feet_width / shoulder_width
        knees_to_feet_ratio = knee_width / (feet_width + 0.0001)
        feet_wide = feet_ratio >= 1.35
        knees_open = knees_to_feet_ratio >= 0.55
        return {
            "angle": angle,
            "left_angle": left_angle,
            "right_angle": right_angle,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_ankle": left_ankle,
            "right_ankle": right_ankle,
            "feet_ratio": feet_ratio,
            "knees_to_feet_ratio": knees_to_feet_ratio,
            "feet_wide": feet_wide,
            "knees_open": knees_open,
            "is_sumo": feet_wide and knees_open
        }

    def _handle_calibration(self, image, metrics):
        angle = metrics["angle"]
        self.view.update_target(f"KAL {self.calibration_count}/{self.calibration_reps}")
        if not metrics["is_sumo"]:
            self.stage = "blad"
            self.view.update_status(self._get_form_feedback(metrics))
            return
        if angle < self.DOWN_CALIBRATION_LIMIT:
            self.stage = "dol"
            self.current_min_angle = min(self.current_min_angle, angle)
            self.view.update_status("Kalibracja: wróć do pozycji stojącej po zejściu w dół.")
        elif angle > self.UP_THRESHOLD and self.stage == "dol":
            if self.current_min_angle < self.DOWN_CALIBRATION_LIMIT:
                self.calibration_angles.append(self.current_min_angle)
                self.calibration_count += 1
                self.audio.speak(f"Kalibracja {self.calibration_count}")
            self.current_min_angle = 180.0
            self.stage = "gora"
            if self.calibration_count >= self.calibration_reps:
                avg_depth = sum(self.calibration_angles) / len(self.calibration_angles)
                self.target_depth = min(max(avg_depth + self.DEPTH_TOLERANCE, self.MIN_TARGET_DEPTH), self.MAX_TARGET_DEPTH)
                self.calibration_done = True
                self.view.update_target(str(int(self.target_depth)))
                self.view.update_status("Kalibracja zakończona. Teraz liczę poprawne powtórzenia.")
                self.audio.speak("Kalibracja zakończona")
            else:
                self.view.update_status(f"Kalibracja: {self.calibration_count}/{self.calibration_reps}. Wykonaj kolejne powtórzenie.")
        elif angle > self.UP_THRESHOLD:
            self.stage = "gora"
            self.view.update_status("Kalibracja: wykonaj spokojny przysiad sumo.")

    def _handle_training(self, image, metrics):
        angle = metrics["angle"]
        self.view.update_target(str(int(self.target_depth)))
        if not metrics["is_sumo"]:
            self.stage = "blad"
            feedback = self._get_form_feedback(metrics)
            self.view.update_status(feedback)
            self.audio.speak(feedback)
            return
        if angle > self.UP_THRESHOLD and self.stage in [None, "blad"]:
            self.stage = "gora"
        if self.stage == "gora" and angle < self.target_depth:
            self.stage = "dol"
            self.view.update_status("Dół zaliczony. Wróć do pozycji stojącej.")
        elif self.stage == "dol" and angle > self.UP_THRESHOLD:
            self.stage = "gora"
            self.counter += 1
            self.current_reps += 1
            self.view.update_counter(str(self.counter))
            self.view.update_status("Poprawne powtórzenie.")
            self.audio.speak(str(self.counter))
        elif self.stage == "gora" and angle < self.target_depth + 15:
            self.view.update_status("Zejdź trochę niżej.")
        else:
            self.view.update_status("Postawa poprawna. Kontynuuj ruch.")

    def _get_form_feedback(self, metrics):
        if not metrics["feet_wide"]:
            return "Ustaw stopy szerzej."
        if not metrics["knees_open"]:
            return "Prowadź kolana bardziej na zewnątrz."
        return "Popraw postawę."

    def _draw_feedback(self, image, metrics, results):
        color = (0, 255, 0) if metrics["is_sumo"] else (0, 0, 255)
        status = "SUMO OK" if metrics["is_sumo"] else "POPRAW POSTAWE"
        cv2.putText(image, status, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.putText(image, f"Angle: {int(metrics['angle'])}", (30, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        cv2.putText(image, f"Feet: {metrics['feet_ratio']:.2f}", (30, 135), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        cv2.putText(image, f"Knees: {metrics['knees_to_feet_ratio']:.2f}", (30, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        self.vision.draw_protractor(image, metrics["left_hip"], metrics["left_knee"], metrics["left_ankle"], metrics["left_angle"], color)
        self.vision.draw_protractor(image, metrics["right_hip"], metrics["right_knee"], metrics["right_ankle"], metrics["right_angle"], color)
        self.vision.draw_landmarks(image, results.pose_landmarks)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    root.configure(fg_color="#242424")
    app = PersonalTrainerApp(root)
    root.mainloop()
