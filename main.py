import gestures
from stt import STT
import threading
import robosuite as suite
from policies import *

class RoboSuite:
    def __init__(self):
        self.env = None
        self.obs = None
        self.policy = None

    def _create_environment(self):
        self.env = suite.make(
            env_name="Stack", # replace with other tasks "Stack" and "Door"
            robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            ignore_done=True,
            render_camera=None,
        )

    def _create_policy(self):
        self.obs = self.env.reset()
        self.policy = StackPolicy(self.obs)

    def _run_environment(self, stt_obj, gesture_recognizer, move_dir=None):
        
        gripper_action = -1
        while True:
            gripper_dir = stt_obj.get_direction()  # Read value for rotation & gripper close
            move_dir = gesture_recognizer.get_direction()  # Read value for movement
            action = self.policy.get_action(self.obs, move_dir, gripper_dir, gripper_action)
            gripper_action = action[-1]
            self.obs, reward, done, info = self.env.step(action)  # take action in the environment

            self.env.render()  # render on display
            if reward == 1.0:
                print("SUCCESSFUL")
                break



if __name__ == "__main__":
    robo_suite = RoboSuite()

    stt = STT()
    #set show_UI to True to see the hand recognition UI on windows
    gesture_recognizer = gestures.GestureRecognizer(show_UI=False)
    
    robo_suite._create_environment()
    robo_suite._create_policy()

    t1 = threading.Thread(target=stt.run_stt, daemon=True)
    t1.start()

    t2 = threading.Thread(target=gesture_recognizer._run_hand_recognition, daemon=True)
    t2.start()
    
    robo_suite._run_environment(stt, gesture_recognizer)

    