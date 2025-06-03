import cv2
import numpy as np
import threading
import robosuite as suite
from policies import *

# create environment instance
env = suite.make(
    env_name="Stack", # replace with other tasks "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

move_dir = None

def draw_landmarks_on_image(rgb_image, detection_result):
    global move_dir

    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    move_dir = compute_direction(hand_landmarks_list)

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

def compute_direction(hand_landmarks):
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
        return
    
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
        
# Options for the hand landmarker
base_options = BaseOptions(model_asset_path='hand_landmarker.task')  # Ensure this path is correct and points to a .tflite file
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.1,
    min_tracking_confidence=0.1,
    running_mode=RunningMode.IMAGE
)
detector = HandLandmarker.create_from_options(options)

# Setup camera capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open capture device")
    exit(1)

obs = env.reset()
policy = StackPolicy(obs)

# First thread is going to do the hand recognition part
def hand_recognition():
    # Run inference on the video
    print("Running hand landmarker...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using HandLandmarker
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        # Draw the hand landmarks on the frame
        if results:
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), results)
            bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Frame", bgr_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

t1 = threading.Thread(target=hand_recognition, daemon=True)
t1.start()

while True:

    # print(move_dir)

    action = policy.get_action(obs, move_dir)
    obs, reward, done, info = env.step(action)  # take action in the environment

    env.render()  # render on display
    if reward == 1.0:
        print("SUCCESSFUL")
        break