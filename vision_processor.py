import cv2
import numpy as np
import math
# --- POPRAWKA IMPORTU MEDIAPIPE ---
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError):
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing


class VisionProcessor:
    """Klasa do analizy postawy, matematyki w 3D i nakładania AR na obraz."""

    def __init__(self):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing

        # Inicjalizacja filtru EMA do wygładzania kąta (usuwa drgania)
        self.ema_memory = {}
        self.smoothed_angle = None
        self.alpha = 0.064  # Mniejsza wartość = większe wygładzenie, ale minimalne opóźnienie

    def calculate_angle_3d(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle) * (180.0 / np.pi)
        return angle
    def get_3d_dist(self, lm1, lm2):

        return math.sqrt(
            (lm1.x - lm2.x) ** 2 +
            (lm1.y - lm2.y) ** 2 +
            (lm1.z - lm2.z) ** 2
        )
    def get_smoothed_angle(self, current_angle):
        """Filtr dolnoprzepustowy (EMA) wygładzający skoki odczytów kamery."""
        if self.smoothed_angle is None:
            self.smoothed_angle = current_angle
        else:
            self.smoothed_angle = (self.alpha * current_angle) + ((1 - self.alpha) * self.smoothed_angle)
        return self.smoothed_angle

    def get_smoothed_value(self, key, current_value, custom_alpha=None):
        alpha = custom_alpha if custom_alpha is not None else self.alpha

        if key not in self.ema_memory:
            self.ema_memory[key] = current_value
        else:
            # St = (alpha * Xt) + ((1 - alpha) * St-1)
            self.ema_memory[key] = (alpha * current_value) + ((1 - alpha) * self.ema_memory[key])

        return self.ema_memory[key]
    def check_visibility(self, landmarks, threshold=0.5):
        """Sprawdza, czy kluczowe stopy i biodra są fizycznie w kadrze kamery."""
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,  # Poprawione: dodany przecinek
            self.mp_pose.PoseLandmark.LEFT_HEEL.value,  # Poprawione: .value oraz przecinek
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,  # Poprawione: pełna ścieżka do enuma
            self.mp_pose.PoseLandmark.RIGHT_HEEL.value,  # Poprawione: pełna ścieżka do enuma
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value  # Poprawione: pełna ścieżka do enuma
        ]

        for point in key_points:
            # Jeśli pewność modelu co do punktu jest mniejsza niż 50%, odrzucamy
            if landmarks[point].visibility < threshold:
                return False
        return True

    @staticmethod
    def draw_protractor(image, a, b, c, angle, color):
        """Rysuje kątomierz w 2D (rzutuje punkty z powrotem na płaski ekran)."""
        h, w, _ = image.shape

        # Nawet jeśli a, b, c mają 3 wymiary (x,y,z), bierzemy tylko indeksy [0] i [1]
        pa = (int(a[0] * w), int(a[1] * h))
        pb = (int(b[0] * w), int(b[1] * h))
        pc = (int(c[0] * w), int(c[1] * h))

        cv2.line(image, pa, pb, color, 4)
        cv2.line(image, pb, pc, color, 4)
        cv2.circle(image, pb, 15, color, -1)
        cv2.circle(image, pb, 20, (255, 255, 255), 2)
        cv2.putText(image, str(int(angle)), (pb[0] + 20, pb[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_pose.POSE_CONNECTIONS)