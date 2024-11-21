# ji_dog_env_create.py
# import modules


from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.physx import get_physx_interface
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import ArticulationView
from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
from omni.usd import get_context

import numpy as np
import omni.usd
import asyncio
import carb
import math
import gym
from omni.isaac.core.objects import DynamicCuboid
import numpy as np


class Ji_Dog_Env(gym.Env):
    def __init__(self, usd_path) -> None:
        super(Ji_Dog_Env, self).__init__()

        # Robot parameters
        self.robot = None
        self.joint_view = None
        self.robot_position = None
        self.robot_orientation = None
        self.robot_linear_velocity = None
        self.robot_angular_velocity = None
        self.joint_positions = None
        self.joint_velocities = None
        self.contact_state = None
        self.usd_path = usd_path

        # Set up scene
        self.slope_flag = False
        self.gait_flag = False
        self.setup_scene()

        # Gait parameters
        self.A = np.radians(30)  # Amplitude, maximum 30 degrees
        self.omega = 2 * np.pi  # Frequency, controlling the speed of gait
        self.phase_shift = np.pi  # Phase shift to switch leg movements
        self.time_period = 1.0  # Duration of each cycle

        # Step parameters
        self.first_step = True
        self.needs_reset = False

        # Define action and observation spaces for reinforcement learning
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

    def setup_scene(self):
        self._world = World()

        # print("PhysicsScene added.")
        # Add ground
        self._world.scene.add_default_ground_plane()

        add_reference_to_stage(
            usd_path= self.usd_path,
            prim_path="/World/Robot",
        )
        self.robot = Robot(prim_path="/World/Robot", name="my_robot")
        if self.robot is None:
            print("Robot model not found, please check the model path and format.")
            return

        self._world.scene.get_object("my_robot")

        self.joint_view = ArticulationView(
            prim_paths_expr="/World/Robot", name="joint_view"
        )
        if not self.joint_view.is_valid():
            print(
                "ArticulationView initialization failed, please check the joint path in the model."
            )
            return

        self._world.scene.add(self.joint_view)
        self.joint_controller = self.robot.get_articulation_controller()

        self.physx_interface = get_physx_interface()
        self.contact_data = {
            "lleg1": True,
            "lleg2": True,
            "lleg3": True,
            "lleg4": True,
        }
        # Use physics step event
        self.physx_interface.subscribe_physics_on_step_events(
            self.on_step, pre_step=False, order=0
        )
        self._world.scene.add(self.robot)

        if self.robot is None:
            print("Failed to find robot articulation.")
        else:
            print("Robot articulation successfully loaded:", self.robot)

        # print(
        #     "Num of degrees of freedom before first reset: " + str(self.robot.num_dof)
        # )

        if self.slope_flag:
            self.setup_slope()

        return

    def setup_slope(self):
        self.slope = DynamicCuboid(
            prim_path="/World/Slope",
            name="slope",
            position=[0, 0, 0],
            size=1.0,
            mass=100,
        )
        self.slope.set_local_scale([20.0, 10.0, 0.2])
        self._world.scene.get_object("slope")
        self._world.scene.add(self.slope)

    def calculate_quaternion(self, roll, pitch, yaw):
        """Calculate Quaternion from RPY"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qw, qx, qy, qz])

    def setup(self) -> None:
        """Set up physics callback"""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        # self._world.add_physics_callback("robot_advance", callback_fn=self.step)

    def generate_alternating_gait(self):
        """Generate alternating gait movements"""
        # Current time of the cycle
        phase = (self.t // self.time_period) % 2  # Cycle control for alternating movement
        action = []

        if phase == 0:
            # Phase 1: Move left front leg and right hind leg, keep right front leg and left hind leg still
            left_front = self.A * np.sin(self.omega * self.t)  # Move left front leg
            right_hind = self.A * np.sin(self.omega * self.t)  # Move right hind leg
            right_front = 0  # Keep right front leg still
            left_hind = 0  # Keep left hind leg still
        else:
            # Phase 2: Move right front leg and left hind leg, keep left front leg and right hind leg still
            left_front = 0  # Keep left front leg still
            right_hind = 0  # Keep right hind leg still
            right_front = self.A * np.sin(self.omega * self.t)  # Move right front leg
            left_hind = self.A * np.sin(self.omega * self.t)  # Move left hind leg

        # Add actions to the list
        action = [left_front, right_hind, right_front, left_hind]

        # Limit action range within a reasonable range (e.g., -45 to 45 degrees)
        action = np.clip(action, np.radians(-45), np.radians(45))
        return np.array(action)

    def get_joint_states(self):
        """Get the angles and velocities of all 4 joints"""

        # Get the current joint angles (positions) and velocities via articulation_view
        joint_positions = self.joint_view.get_joint_positions()
        joint_velocities = self.joint_view.get_joint_velocities()

        return joint_positions, joint_velocities

    def on_step(self):

        leg_names = ["leg1", "leg2", "leg3", "leg4"]
        for i, leg_name in enumerate(leg_names):

            is_contacting = self.is_leg_in_contact(i)
            # print("is_contacting", is_contacting)
            self.contact_data[leg_name] = is_contacting

        # Print contact information for debugging
        # print(f"Contact status: {self.contact_data}")

        leg_contact_states = [
            self.contact_data[leg_name] for leg_name in ["leg1", "leg2", "leg3", "leg4"]
        ]
        # print("Contact", leg_contact_states)

        return tuple(leg_contact_states)

    def is_leg_in_contact(self, leg_index):
        """Simulate a method that checks if a leg is in contact with the ground"""
        contact_threshold = 0.5

        return abs(self.joint_positions[0][leg_index]) < contact_threshold

    def get_observation(self):
        # The state of body
        # x, y, z: Position of the center of mass of the agent in 3D space.
        self.robot_position, robot_orientation0 = self.robot.get_world_pose()

        # v_x, v_y, v_z: Translational velocities of the center of mass along the x, y, and z axes.
        self.robot_linear_velocity = self.robot.get_linear_velocity()

        # phi, theta, psi: Euler angles representing the orientation (roll, pitch, yaw) of the agent.
        self.robot_orientation = self.calculate_rpy(robot_orientation0)

        # omega_x, omega_y, omega_z: Angular velocities of the center of mass around the x, y, and z axes.
        self.robot_angular_velocity = self.robot.get_angular_velocity()

        # The state of joint
        # theta_1, theta_2, theta_3, theta_4: Joint angles of all 4 joints.
        self.joint_positions, self.joint_velocities = self.get_joint_states()

        # Print joint states
        # print(f"Joint positions: {self.joint_positions}")
        # print(f"Joint velocities: {self.joint_velocities}")
        # $(c_1, c_2, c_3, c_4)$: The state of each foot's contact with the ground (Boolean values) where $c_i=1$ indicates that the $i$-th foot is in contact with the ground and $c_i=0$ indicates no contact.
        self.contact_state = self.on_step()

        observation = {
            "Robot Position": self.robot_position,
            "Robot Orientation": self.robot_orientation,
            "Robot Linear Velocity": self.robot_linear_velocity,
            "Robot Angular Velocity": self.robot_angular_velocity,
            "Joint positions": self.joint_positions,
            "Joint velocities": self.joint_velocities,
        }

        return observation

    def calculate_reward(self):
        # pos, ori are position and orientation of the mass centre of the dog
        # cont is the contact indicator of the dogs
        reward = 0 
        pos = self.robot_position
        ori = self.robot_orientation
        cont = self.contact_state

        [x_Goal, y_Goal, z_Goal] = self.goal  # x,y position of goal
        Goal_reward = 50  # define Goal reward
        Fall_penalty = -10  # define Fall penalty
        Contact_penalty = -1  # define Contact Point Penalty
        r_allow = 10  # degree unit
        p_allow = 10  # degree unit

        px = pos[0]
        py = pos[1]
        pz = pos[2]
        [r, p, y] = ori

        # Contact point definition according to the professor
        L1 = cont[0]  # left legs
        L2 = cont[1]
        R1 = cont[2]  # right legs
        R2 = cont[3]

        k2 = 10
        k3 = 10
        k5 = 10

        # Goal reward
        if px == x_Goal and py == y_Goal:
            reward += Goal_reward

        # Fall Penalty
        if pz == 0:
            reward += Fall_penalty

        # Contact Point Penalty
        if [L1, L2, R1, R2] == [1, 1, 1, 1] or [L1, L2] == [0, 0] or [R1, R2] == [0, 0]:
            reward += Contact_penalty

        # Progress reward
        # reward += math.exp(-(abs(px - x_Goal) + abs(py - y_Goal)))
        reward += k2 * math.exp(-(abs(px-x_Goal) + abs(py-y_Goal))) # an additional coefficient can be added

        # Mass Centre reward
        # reward += math.exp(-abs(pz - z_Goal))
        reward += k3 * math.exp(-abs(pz-z_Goal))  # an additional coefficient can be added

        # Stability penalty
        # reward += min(0, r_allow - abs(r)) + min(0, p_allow - abs(p))
        reward += k5 * (min(0, r_allow - abs(r)) + min(0, p_allow - abs(p)))  # an additional coefficient can be added

        return reward

    def calculate_rpy(self, Quaternion):  # convert quaternion to roll pitch yaw (r, p, y)

        # Quaternion
        qw = Quaternion[0]
        qx = Quaternion[1]
        qy = Quaternion[2]
        qz = Quaternion[3]

        # get roll pitch yaw in radians
        r = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        # p = math.asin(2 * (qw * qy - qz * qx))
        p = math.asin(
            max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
        )  # avoid exceeding range of asin
        y = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        # print([r,p,y])  # for testing only

        # get roll pitch yaw in degrees
        r = math.degrees(r)
        p = math.degrees(p)
        y = math.degrees(y)

        return [r, p, y]

    def step(self, setp_action) -> None:

        self.set_goal()

        if self.first_step:
            self.robot.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            if self.gait_flag:
                action = self.generate_alternating_gait()
            else:
                action = setp_action

            # asyncio.run(self.control_joints_async(action))
            action = ArticulationAction(
                joint_positions = action,
                joint_efforts=None,
                joint_velocities= None,
            )
            # print("move", action)
            self.joint_controller.apply_action(action)


        # Take a simulation step
        self._world.step(render=True)

        # Get sensor data
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if done
        done = self.is_done()

        return observation, reward, done, {}

    def set_goal(self):
        self.goal = [10.0, 10.0, 0.5]
        return

    def is_done(self):
        if self.robot_position[0:1] == self.goal[0:1]:
            return True
        else:
            return False

    def reset(self):
        self._world.reset()
        observation = self.get_observation()
        return observation

    def close(self):
        simulation_app.close()


def main():
    # Initialize the environment
    dog = Ji_Dog_Env('/home/bai/.local/share/ov/pkg/isaac-sim-4.2.0/Ji-dog 2.0/Model(including video)/ji_dog.usd')

    simulation_app.update()
    dog.setup()
    simulation_app.update()
    dog.reset()
    simulation_app.update()

    # Define the total number of steps to run
    num_steps = 50000
    dog.t = 0
    dt = 0.05

    # Assume we have a random action generator
    for step in range(num_steps):
        # Perform one simulation step, return observation, reward, done status, and additional info
        observation, reward, done, info = dog.step(dog.t)

        dog.t += dt

        # Print the information of this simulation step
        # print(f"Step: {step}")
        # print(f"Observation: {observation}")
        # print(f"Reward: {reward}")
        # print(f"Done: {done}")

        # Exit if the simulation is done
        if done:
            break

    # Close the environment after the simulation is complete
    dog.close()


# # Entry point of the main function
# if __name__ == "__main__":
#     main()
