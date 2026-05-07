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
    """Klasa do analizy postawy, matematyki i nakładania AR na obraz."""
    def __init__(self):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing

    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    @staticmethod
    def draw_protractor(image, a, b, c, angle, color):
        h, w, _ = image.shape
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