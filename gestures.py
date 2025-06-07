import numpy as np
import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class GestureRecognizer:
    def __init__(self, show_UI=False):
        self.base_options = BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = HandLandmarkerOptions(
            base_options=self.base_options,
            num_hands=2,
            min_hand_detection_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        self.detector = HandLandmarker.create_from_options(self.options)
        self.move_dir = None
        self.cap = None
        self.show_UI = show_UI

    def get_direction(self):
        return self.move_dir

    def _draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        self.move_dir = self._compute_direction(hand_landmarks_list)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

        return annotated_image

    def _compute_direction(self, hand_landmarks):
        if len(hand_landmarks) == 0:
            return None

        hand_landmarks = hand_landmarks[0]

        top_thumb = np.array([hand_landmarks[4].x, hand_landmarks[4].y, hand_landmarks[4].z])
        bot_thumb = np.array([hand_landmarks[3].x, hand_landmarks[3].y, hand_landmarks[3].z])

        top_index = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
        bot_index = np.array([hand_landmarks[7].x, hand_landmarks[7].y, hand_landmarks[7].z])

        thumb = top_thumb - bot_thumb
        index = top_index - bot_index
        if np.dot(thumb, index) >= 0.0005:
            return None

        x_axis = np.array([1, 0])
        v = (top_index - bot_index)[:-1]
        angle = np.arccos(np.dot(x_axis, v) / np.linalg.norm(v)) * 180 / np.pi

        if index[1] >= 0 and (70 < angle and angle < 120):
            return "DOWN"
        elif index[1] < 0 and (70 < angle and angle < 120):
            return "UP"
        elif index[0] > 0 and  (0 <= angle and angle < 20):
            return "RIGHT"
        elif index[0] < 0 and  (160 <= angle and angle < 180):
            return "LEFT"
        else:
            return None

    def _add_ui_elements(self, frame: np.ndarray, directions) -> np.ndarray:
        """Add UI elements to the frame."""
        # Add background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add gesture text
        if directions:
            cv2.putText(frame, f"Directions: {directions}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No gesture detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' or ESC to quit", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    

    def _hand_recognition(self):
        # Run inference on the video
        print("Running hand landmarker...")

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame using HandLandmarker
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)

            # Draw the hand landmarks on the frame
            if results:
                hand_landmarks_list = results.hand_landmarks
                self.move_dir = self._compute_direction(hand_landmarks_list)
                if self.show_UI:
                    annotated_image = self._draw_landmarks_on_image(mp_image.numpy_view(), results)
                    annotated_image = self._add_ui_elements(annotated_image, self.move_dir)
                    bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Frame", bgr_frame)
                

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

        
        self.cap.release()
        cv2.destroyAllWindows()

    def _run_hand_recognition(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Failed to open capture device")
            exit(1)

        else:
            self._hand_recognition()

if __name__ == "__main__":
    gesture_recognizer = GestureRecognizer()
    gesture_recognizer._run_hand_recognition()