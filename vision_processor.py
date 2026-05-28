import cv2
import numpy as np
import math

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
        self.ema_memory = {}
        self.alpha = 0.18

    def calculate_angle_3d(self, a, b, c):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)
        ba = a - b
        bc = c - b
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0:
            return 180.0
        cosine_angle = np.dot(ba, bc) / norm
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))

    def get_3d_dist(self, lm1, lm2):
        return math.sqrt(
            (lm1.x - lm2.x) ** 2 +
            (lm1.y - lm2.y) ** 2 +
            (lm1.z - lm2.z) ** 2
        )

    def get_smoothed_value(self, key, current_value, custom_alpha=None):
        alpha = custom_alpha if custom_alpha is not None else self.alpha
        if key not in self.ema_memory:
            self.ema_memory[key] = current_value
        else:
            self.ema_memory[key] = alpha * current_value + (1 - alpha) * self.ema_memory[key]
        return self.ema_memory[key]

    def reset_smoothing(self):
        self.ema_memory.clear()

    def check_visibility(self, landmarks, threshold=0.5):
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ]
        return all(landmarks[point].visibility >= threshold for point in key_points)

    @staticmethod
    def landmark_to_point(landmark):
        return [landmark.x, landmark.y, landmark.z]

    @staticmethod
    def draw_protractor(image, a, b, c, angle, color):
        h, w, _ = image.shape
        pa = (int(a[0] * w), int(a[1] * h))
        pb = (int(b[0] * w), int(b[1] * h))
        pc = (int(c[0] * w), int(c[1] * h))
        cv2.line(image, pa, pb, color, 4)
        cv2.line(image, pb, pc, color, 4)
        cv2.circle(image, pb, 12, color, -1)
        cv2.circle(image, pb, 18, (255, 255, 255), 2)
        cv2.putText(image, str(int(angle)), (pb[0] + 20, pb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_pose.POSE_CONNECTIONS)
