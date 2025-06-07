import numpy as np
import time
from scipy.spatial.transform import Rotation as R

import robosuite as suite
from pid import PID

class StackPolicy(object):
    """
    PID controller to stack cubes on top of one another
    """
    def __init__(self, obs):
        """
        Initialize the TouchPolicy with the first observation from the environment.
        Args:
            obs (dict): Initial observation from the environment. Must include:
                - 'cube_pos': The position of the cube to be touched.
        """
        self.cubeA_pos = obs["cubeA_pos"]
        self.cubeB_pos = obs["cubeB_pos"]

        self.stage = 0
        self.stages = [self.cubeA_pos.copy(),   # Go to Cube A + dZ
                       self.cubeA_pos.copy(),   # Go to Cube A
                       self.cubeB_pos.copy(),   # Go to Cube B + dZ
                       self.cubeB_pos.copy(),]  # Go to Cube B
        
        kp = 1
        ki = 3
        kd = 2
        target_pos = self.cubeA_pos
        target_pos[2] += 0.2

        self.pid = PID(kp, ki, kd, target_pos)

        self.prev_time = time.time_ns() / 1e6
        self.dt = 50 # ms

        self.prev_ctrl_output = np.array([0, 0, 0])

        self.in_progress = False

        # SpeedConstants
        self.GRANULARITY = 0.001
        self.FORWARD_GRANULARITY = self.BACKWARD_GRANULARITY = 0.0005
        self.ROTATION_GRANULARITY = 0.01
    
    def get_action(self, obs, direction=None, gripper_dir=None, gripper_action=None):
        
        OBJ_DIST_THRESH = 0.01

        curr_time = time.time_ns() / 1e6

        eef_pos = obs["robot0_eef_pos"]
        control = None
        rotation = np.zeros(3)
        
            
        if self.in_progress:
            dist = np.linalg.norm(self.pid.target - eef_pos)
            
            if dist < OBJ_DIST_THRESH:
                self.pid.reset(target=None)
                self.in_progress = False
                
                control = np.zeros(3)
            else:
                control = self.pid.update(eef_pos)
        elif direction:
            # print("CHANGIN DIRECTION:", direction)
            match direction: #matches hand gestures
                case "UP":
                    dir = np.array([0, 0, 1])
                    self.pid.reset(target=eef_pos.copy() + self.GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "DOWN":
                    dir = np.array([0, 0, -1])
                    self.pid.reset(target=eef_pos.copy() + self.GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "RIGHT":
                    dir = np.array([0, 1, 0])
                    self.pid.reset(target=eef_pos.copy() + self.GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "LEFT":
                    dir = np.array([0, -1, 0])
                    self.pid.reset(target=eef_pos.copy() + self.GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case _:
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0])
                    self.in_progress = False        
        else:
            match gripper_dir: #matches voice commands
                case "CLOSE":
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0])
                    gripper_action = 1

                case "OPEN":
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0])
                    gripper_action = -1

                case "ROTATE_R":
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0.01])

                case "ROTATE_L":
                    control = np.zeros(3)
                    rotation = np.array([0, 0, -0.01])

                case "SPEED_UP":
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0])
                    self.in_progress = False
                    self.GRANULARITY += 0.002
                    self.FORWARD_GRANULARITY += 0.0002
                    self.BACKWARD_GRANULARITY += 0.0002
                    self.ROTATION_GRANULARITY += 0.02

                case "SLOW_DOWN":
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0])
                    self.in_progress = False
                    self.GRANULARITY -= 0.002
                    self.FORWARD_GRANULARITY -= 0.0002
                    self.BACKWARD_GRANULARITY -= 0.0002
                    self.ROTATION_GRANULARITY -= 0.02

                case "RESET":
                    self.GRANULARITY = 0.001
                    self.FORWARD_GRANULARITY = self.BACKWARD_GRANULARITY = 0.0005
                    self.ROTATION_GRANULARITY = 0.01
                
                case "FORWARD":
                    if not self.in_progress:
                        dir = np.array([1, 0, 0])
                        self.pid.reset(target=eef_pos.copy() + self.FORWARD_GRANULARITY * dir)
                        self.in_progress = True
                        control = self.pid.update(eef_pos)

                case "BACKWARD":
                    if not self.in_progress:
                        dir = np.array([-1, 0, 0])
                        self.pid.reset(target=eef_pos.copy() + self.BACKWARD_GRANULARITY * dir)
                        self.in_progress = True
                        control = self.pid.update(eef_pos)

                case "STOP":
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0])
                    self.in_progress = False
                    gripper_dir = ""
                
                case _:
                    control = np.zeros(3)
                    rotation = np.array([0, 0, 0])
                    self.in_progress = False
        
        return np.concatenate([control, rotation, [gripper_action]])