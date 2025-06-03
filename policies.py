import numpy as np
import time
from scipy.spatial.transform import Rotation as R

import robosuite as suite
from pid import PID

class LiftPolicy(object):
    """
    A simple P-controller policy for a robotic arm to touch an object in two phases:
    1. Move directly to the object.
    2. Then, hover above the object.
    We only need a proportional controller to drive the robot's end-effector!
    """
    def __init__(self, obs):
        """
        Initialize the TouchPolicy with the first observation from the environment.
        Args:
            obs (dict): Initial observation from the environment. Must include:
                - 'cube_pos': The position of the cube to be touched.
        """
        self.cube_pos = obs["cube_pos"]
        self.stage = 0
        
        kp = 2.0
        ki = 0.0
        kd = 0.0
        target_pos = self.cube_pos

        self.pcontroller = PID(kp, ki, kd, target_pos)

        self.prev_time = time.time_ns() / 1e6
        self.dt = 200 # ms

        self.prev_ctrl_output = np.array([0, 0, 0])

    def get_action(self, obs):
        """
        Compute the next action for the robot based on current observation.

        Args:
            obs (dict): Current observation. Must include:
                - 'robot0_eef_pos': Current end-effector position.
                - 'cube_pos': Current position of the cube.

        Returns:
            np.ndarray: 7D action array for robosuite OSC:
                - action[-1]: Gripper command (1 to close, -1 to open)
        """
        # Set some constants to play around with
        OBJ_DIST_THRESH = 0.003
        HOVER_DIST_THRESH = 0.1

        # Get current time
        curr_time = time.time_ns() / 1e6

        eef_pos = obs["robot0_eef_pos"]
        current_cube_pos = obs["cube_pos"]

        open_grip = False

        if self.stage == 0:
            # Just move directly towards the object.
            target_pos = self.cube_pos
            # Technically, we don't need reset here. 
            # self.pcontroller.reset(target_pos)

            dist = np.linalg.norm(self.pcontroller.target - eef_pos)

            if dist < OBJ_DIST_THRESH:
                self.stage = 1 # Update to our new stage.
                # print("Moved to stage 1")

            open_grip = True
        elif self.stage == 1:
            self.stage = 2
        elif self.stage == 2:
            hover_pos = current_cube_pos.copy()
            hover_pos[2] += HOVER_DIST_THRESH
            # Q: for students: why do we do .reset()? What happens if we leave it out?
            self.pcontroller.reset(hover_pos)

        ctrl_output = self.prev_ctrl_output
        if curr_time - self.prev_time >= self.dt:
            ctrl_output = self.pcontroller.update(eef_pos, self.dt)
            self.prev_time = curr_time

        # Q: for students: What is the action space of the robot?
        action = np.zeros(7)
        # Q: for students: Why do we only set the first 3 values?
        action[0:3] = ctrl_output
        # For opening gripper
        action[-1] = -1 if open_grip else 1

        self.prev_ctrl_output = ctrl_output

        return action

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
        
        kp = 4
        ki = 1
        kd = 1
        target_pos = self.cubeA_pos
        target_pos[2] += 0.2

        self.pid = PID(kp, ki, kd, target_pos)

        self.prev_time = time.time_ns() / 1e6
        self.dt = 50 # ms

        self.prev_ctrl_output = np.array([0, 0, 0])

        self.in_progress = False
    
    def get_action(self, obs, direction=None):
        # Constants
        GRANULARITY = 0.01
        OBJ_DIST_THRESH = 0.01

        curr_time = time.time_ns() / 1e6

        eef_pos = obs["robot0_eef_pos"]

        control = None
        if self.in_progress:
            dist = np.linalg.norm(self.pid.target - eef_pos)
            
            if dist < OBJ_DIST_THRESH:
                self.pid.reset(target=None)
                self.in_progress = False
                
                control = np.zeros(3)
            else:
                control = self.pid.update(eef_pos)
        else:
            print("CHANGIN DIRECTION:", direction)
            match direction:
                case "UP":
                    dir = np.array([0, 0, 1])
                    self.pid.reset(target=eef_pos.copy() + GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "DOWN":
                    dir = np.array([0, 0, -1])
                    self.pid.reset(target=eef_pos.copy() + GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "RIGHT":
                    dir = np.array([0, 1, 0])
                    self.pid.reset(target=eef_pos.copy() + GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "LEFT":
                    dir = np.array([0, -1, 0])
                    self.pid.reset(target=eef_pos.copy() + GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "FORWARD":
                    dir = np.array([1, 0, 0])
                    self.pid.reset(target=eef_pos.copy() + GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case "BACKWARD":
                    dir = np.array([-1, 0, 0])
                    self.pid.reset(target=eef_pos.copy() + GRANULARITY * dir)
                    self.in_progress = True

                    control = self.pid.update(eef_pos)
                case _:
                    control = np.zeros(3)
                    self.in_progress = False
            
        return np.concatenate([control, np.zeros(3), [-1]])


                
            


    def get_action_original(self, obs):
        """
        Compute the next action for the robot based on current observation.

        Args:
            obs (dict): Current observation. Must include:
                - 'robot0_eef_pos': Current end-effector position.
                - 'cubeA_pos': Current position of cube A.
                - 'cubeB_pos': Current position of cube B.

        Returns:
            np.ndarray: 7D action array for robosuite OSC:
                - action[-1]: Gripper command (1 to close, -1 to open)
        """
        # Update stages positions
        self.update_stages(obs)

        # Set some constants to play around with
        OBJ_DIST_THRESH = 0.01
        HOVER_DIST_THRESH = 0.1

        # Get current time
        curr_time = time.time_ns() / 1e6

        eef_pos = obs["robot0_eef_pos"]

        open_grip = False

        delta_quat = np.array([0,0,0])

        if self.stage == 0:
            dist = np.linalg.norm(self.pcontroller.target - eef_pos)

            # open_grip = True

            if dist < OBJ_DIST_THRESH:
                self.stage += 1 # Update to our new stage.
                self.pcontroller.reset(self.stages[self.stage])
                # print("Target:", self.pcontroller.target)
                # print("Stages:", self.stages)
                # print("Moved to stage", self.stage)
            else:
                delta_yaw = 0.25  # radians
                r = R.from_euler('z', delta_yaw)
                delta_quat = r.as_quat()
        elif self.stage == 1:
            dist = np.linalg.norm(self.pcontroller.target - eef_pos)

            open_grip = True

            if dist < OBJ_DIST_THRESH:
                self.stage += 1 # Update to our new stage.
                self.pcontroller.reset(self.stages[self.stage])
                open_grip = False
                # print("Target:", self.pcontroller.target)
                # print("Stages:", self.stages)
                # print("Moved to stage", self.stage)
        elif self.stage == 2:
            dist = np.linalg.norm(self.pcontroller.target - eef_pos)

            if dist < OBJ_DIST_THRESH:
                self.stage += 1 # Update to our new stage.
                self.pcontroller.reset(self.stages[self.stage])
                # print("Target:", self.pcontroller.target)
                # print("Stages:", self.stages)
                # print("Moved to stage", self.stage)

        elif self.stage == 3:
            dist = np.linalg.norm(self.pcontroller.target - eef_pos)

            if dist < OBJ_DIST_THRESH:
                self.stage += 1
                # print("Moved to stage", self.stage)
        elif self.stage == 4:
            open_grip = True

        ctrl_output = self.prev_ctrl_output
        if curr_time - self.prev_time >= self.dt:
            ctrl_output = self.pcontroller.update(eef_pos, self.dt)
            self.prev_time = curr_time

        # Q: for students: What is the action space of the robot?
        action = np.zeros(7)
        # Q: for students: Why do we only set the first 3 values?
        action[0:3] = ctrl_output
        # For opening gripper
        action[-1] = -1 if open_grip else 1

        action[3:6] = delta_quat[0:3]

        self.prev_ctrl_output = ctrl_output

        return action
