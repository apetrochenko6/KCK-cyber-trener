import cv2
import numpy as np

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
        self.smoothed_angle = None
        self.alpha = 0.2  # Mniejsza wartość = większe wygładzenie, ale minimalne opóźnienie

    def calculate_angle_3d(self, a, b, c):
        """Oblicza rzeczywisty kąt 3D korzystając z wektorów w przestrzeni (x, y, z)."""
        a, b, c = np.array(a), np.array(b), np.array(c)

        # Wektory od stawu środkowego (kolana)
        ba = a - b
        bc = c - b

        # Iloczyn skalarny i normy wektorów: cos(theta) = (u * v) / (|u| * |v|)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        # Zabezpieczenie przed błędami precyzji (wartości minimalnie poza -1.0 i 1.0)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def get_smoothed_angle(self, current_angle):
        """Filtr dolnoprzepustowy (EMA) wygładzający skoki odczytów kamery."""
        if self.smoothed_angle is None:
            self.smoothed_angle = current_angle
        else:
            self.smoothed_angle = (self.alpha * current_angle) + ((1 - self.alpha) * self.smoothed_angle)
        return self.smoothed_angle

    def check_visibility(self, landmarks, threshold=0.5):
        """Sprawdza, czy kluczowe stopy i biodra są fizycznie w kadrze kamery."""
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
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