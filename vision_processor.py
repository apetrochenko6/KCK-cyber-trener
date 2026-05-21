import cv2
import numpy as np

try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError):
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

class VisionProcessor:
    def __init__(self):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.smoothed_angle = None
        self.alpha = 0.2

    def calculate_angle_3d(self, a, b, c):
        """Oblicza kąt trójwymiarowy między trzema punktami w przestrzeni."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_angle_2d_vertical(self, a, b):
        """Mierzy kąt odchylenia wektora (np. tułowia) od pionowej osi ekranu Y."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        if dy == 0:
            return 90.0
        angle = np.degrees(np.arctan(abs(dx) / abs(dy)))
        return angle

    def get_smoothed_angle(self, current_angle):
        """Stosuje filtr EMA usuwający drgania współrzędnych dostarczanych przez kamerę."""
        if self.smoothed_angle is None:
            self.smoothed_angle = current_angle
        else:
            self.smoothed_angle = (self.alpha * current_angle) + ((1 - self.alpha) * self.smoothed_angle)
        return self.smoothed_angle

    def check_visibility(self, landmarks, threshold=0.5):
        """Sprawdza czy kluczowe punkty potrzebne do analizy sumo są widoczne w kadrze."""
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ]
        for point in key_points:
            if landmarks[point].visibility < threshold:
                return False
        return True

    @staticmethod
    def draw_protractor(image, a, b, c, angle, color):
        """Nakłada na podgląd wideo graficzną wizualizację szkieletu i kątomierza."""
        h, w, _ = image.shape
        pa = (int(a[0] * w), int(a[1] * h))
        pb = (int(b[0] * w), int(b[1] * h))
        pc = (int(c[0] * w), int(c[1] * h))

        cv2.line(image, pa, pb, color, 4)
        cv2.line(image, pb, pc, color, 4)
        cv2.circle(image, pb, 15, color, -1)
        cv2.circle(image, pb, 20, (255, 255, 255), 2)
        cv2.putText(image, str(int(angle)), (pb[0] + 20, pb[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_landmarks(self, image, landmarks):
        """Rysuje standardową mapę połączeń stawów MediaPipe."""
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_pose.POSE_CONNECTIONS)