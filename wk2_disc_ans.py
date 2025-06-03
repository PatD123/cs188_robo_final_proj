# Week 2 Discussion
import numpy as np
import robosuite as suite

# pid.py
class PController:
    def __init__(self, kp, target):
        """
        Initialize a proportional controller.

        Args:
            kp (float): Proportional gain.
            target (tuple or array): Target position.
        """
        self.kp = kp
        self.target = target

    def reset(self, target=None):
        """
        Reset the target position.

        Args:
            target (tuple or array, optional): New target position.
        """
        self.target = target

    def update(self, current_pos):
        """
        Compute the control signal.

        Args:
            current_pos (array-like): Current position.

        Returns:
            np.ndarray: Control output vector.
        """
        error = self.target - current_pos
        return self.kp * error

# policies.py    
class HoverPolicy(object):
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
        
        kp = 2.0 # A hyperparameter to play around with. What does scaling this value do?
        target_pos = self.cube_pos

        self.pcontroller = PController(kp, target_pos)

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
                print("Moved to stage 1")

            open_grip = True
        elif self.stage == 1:
            self.stage = 2
        elif self.stage == 2:
            hover_pos = current_cube_pos.copy()
            hover_pos[2] += HOVER_DIST_THRESH
            # Q: for students: why do we do .reset()? What happens if we leave it out?
            self.pcontroller.reset(hover_pos)

        ctrl_output = self.pcontroller.update(eef_pos)

        # Q: for students: What is the action space of the robot?
        action = np.zeros(7)
        # Q: for students: Why do we only set the first 3 values?
        action[0:3] = ctrl_output
        # For opening gripper
        action[-1] = -1 if open_grip else 1

        return action
                
# test.py
# Create environment instance
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# Reset the environment
for _ in range(5):
    obs = env.reset()
    policy = HoverPolicy(obs)
    
    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment
        
        env.render()  # render on display
        if reward == 1.0 or done:
            break
