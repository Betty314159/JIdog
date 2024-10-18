# import modules
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import ArticulationView
from omni.physx import get_physx_interface

import carb
import omni.physx

import asyncio
import omni.usd
import numpy as np
import math
import gym

class Ji_Dog_Env(gym.Env):
    def __init__(self) -> None:
        super(Ji_Dog_Env, self).__init__()
        self.setup_scene()

         # Gait parameters
        self.A = np.radians(30)  # Amplitude, maximum 30 degrees
        self.omega = 2 * np.pi  # Frequency, controlling the speed of gait
        self.phase_shift = np.pi  # Phase shift to switch leg movements
        self.time_period = 1.0  # Duration of each cycle
        self.first_step = True
        self.needs_reset = False
        # Define action and observation spaces for reinforcement learning
        # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def setup_scene(self):
        self._world = World()
        
        print("PhysicsScene added.")
        # Add ground
        self._world.scene.add_default_ground_plane()
    
        add_reference_to_stage(usd_path='/home/bai/ji_dog.usd', prim_path="/World/Robot")
        self.robot = Robot(prim_path="/World/Robot", name="my_robot")
        if self.robot is None:
            print("Robot model not found, please check the model path and format.")
            return
        
        self._world.scene.get_object('my_robot')

        self.joint_view = ArticulationView(prim_paths_expr="/World/Robot", name="joint_view")
        if not self.joint_view.is_valid():
            print("ArticulationView initialization failed, please check the joint path in the model.")
            return
        
        self._world.scene.add(self.joint_view)
        self.joint_controller = self.robot.get_articulation_controller()

        self.physx_interface = get_physx_interface()
        self.contact_data = {
            "leg1": False,
            "leg2": False,
            "leg3": False,
            "leg4": False,
        }
        # Use physics step event
        self.physx_interface.subscribe_physics_on_step_events(self.on_step, pre_step=False, order=0)
        self._world.scene.add(self.robot)

        if self.robot is None:
            print("Failed to find robot articulation.")
        else:
            print("Robot articulation successfully loaded:", self.robot)
        
        print("Num of degrees of freedom before first reset: " + str(self.robot.num_dof))
        return
    
    def setup(self) -> None:
        """
        Set up keyboard listener and add physics callback

        """
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        # self._keyboard = self._appwindow.get_keyboard()
        # self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("robot_advance", callback_fn=self.step)

    def generate_alternating_gait(self, t):
        """Generate alternating gait movements"""
        # Current time of the cycle
        phase = (t // self.time_period) % 2  # Cycle control for alternating movement
        action = []

        if phase == 0:
            # Phase 1: Move left front leg and right hind leg, keep right front leg and left hind leg still
            left_front = self.A * np.sin(self.omega * t)  # Move left front leg
            right_hind = self.A * np.sin(self.omega * t)  # Move right hind leg
            right_front = 0  # Keep right front leg still
            left_hind = 0  # Keep left hind leg still
        else:
            # Phase 2: Move right front leg and left hind leg, keep left front leg and right hind leg still
            left_front = 0  # Keep left front leg still
            right_hind = 0  # Keep right hind leg still
            right_front = self.A * np.sin(self.omega * t)  # Move right front leg
            left_hind = self.A * np.sin(self.omega * t)  # Move left hind leg

        # Add actions to the list
        action = [left_front, right_hind, right_front, left_hind]

        # Limit action range within a reasonable range (e.g., -45 to 45 degrees)
        action = np.clip(action, np.radians(-45), np.radians(45))
        return np.array(action)

    async def control_joint(self, joint_idx, position):
        """Control the position of a single joint"""
        if self.robot.num_dof == 0:
            print("The robot has no degrees of freedom, please check the model loading.")
            # return
        
        # Ensure the joint index is within the valid range
        if joint_idx >= self.robot.num_dof:
            print(f"Index {joint_idx} is out of bounds for the number of joints.")
            # return

        position = np.clip(position, -1/6 * np.pi, 1/6 * np.pi)
        print(f"Setting joint {joint_idx} to position {np.degrees(position)} degrees")
        
        joint_positions = np.zeros(4)  # Initialize joint position array
        joint_positions[joint_idx] = position  # Control the specified joint
        print(joint_positions)
        # Create ArticulationAction and apply joint positions to the joint controller
        action = ArticulationAction(
            joint_positions = joint_positions,
            joint_efforts = None,
            joint_velocities = None
            )
        self.joint_controller.apply_action(action)
        if self.joint_controller is None:
            print("Joint controller failed to initialize, please check the joint definitions.")
        await asyncio.sleep(0)  # Simulate an asynchronous task

    async def control_joints_async(self, action):
        """Asynchronously control all four joints"""
        tasks = []
        joint_names = ["joint1", "joint2", "joint3", "joint4"]  # Assume these are the joint names
        for i, joint_name in enumerate(joint_names):
            # Apply action to the joints
            position = action[i]
            tasks.append(self.control_joint(i, position))
        await asyncio.gather(*tasks)

    def get_joint_states(self):
        """Get the angles and velocities of all 4 joints"""

        # Get the current joint angles (positions) and velocities via articulation_view
        joint_positions =  self.joint_view.get_joint_positions()
        joint_velocities =  self.joint_view.get_joint_velocities()

        return joint_positions, joint_velocities

    def on_step(self, step_time: float):
        """Update contact information at each simulation step"""
        print(f"Simulation step time: {step_time}")
        # You can update the contact information here
        leg_names = ["leg1", "leg2", "leg3", "leg4"]
        for leg_name in leg_names:
            # Add logic here to determine if a leg is in contact with the ground
            # Assume a method is_leg_in_contact returns whether contact is made
            is_contacting = self.is_leg_in_contact(leg_name)
            self.contact_data[leg_name] = is_contacting
        
        # Print contact information for debugging
        print(f"Contact status: {self.contact_data}")
    
    def is_leg_in_contact(self, leg_name):
        """Simulate a method that checks if a leg is in contact with the ground"""
        contact_threshold = 1

        return self.physx_interface.get_force(leg_name) > contact_threshold

    def get_contact_state(self):
        """Get the contact status of all four legs with the ground"""
        # Clear the previous contact status
        for leg_name in self.contact_data.keys():
            self.contact_data[leg_name] = False

        leg_contact_states = [self.contact_data[leg_name] for leg_name in ["leg1", "leg2", "leg3", "leg4"]]
        print('Contact', leg_contact_states)
        return tuple(leg_contact_states)
         
    def get_observation(self):
        # body
        # $(x, y, z)$: Position of the center of mass of the agent in 3D space.
        self.robot_position, robot_orientation0 = self.robot.get_world_pose()
        # $(v_x, v_y, v_z)$: Translational velocities of the center of mass along the x, y, and z axes.
        self.robot_linear_velocity = self.robot.get_linear_velocity()
        # $(\phi, \theta, \psi)$: Euler angles representing the orientation (roll, pitch, yaw) of the agent.
        self.robot_orientation = self.calculate_rpy(robot_orientation0)
        # $(\omega_x, \omega_y, \omega_z)$: Angular velocities of the center of mass around the x, y, and z axes.
        self.robot_angular_velocity = self.robot.get_angular_velocity()
        # joint
        # $(\theta_1, \theta_2, \theta_3, \theta_4)$: Joint angles of all 4 joints.
        self.joint_positions, self.joint_velocities = self.get_joint_states()

        # Print joint states
        print(f"Joint positions: {self.joint_positions}")
        print(f"Joint velocities: {self.joint_velocities}")
        # $(c_1, c_2, c_3, c_4)$: The state of each foot's contact with the ground (Boolean values) where $c_i=1$ indicates that the $i$-th foot is in contact with the ground and $c_i=0$ indicates no contact.
        self.contact_state = self.get_contact_state()
        # self.contact_state = [0,0,0,0]
        return 

    def calculate_reward(self):
        # pos, ori are position and orientation of the mass centre of the dog
        # cont is the contact indicator of the dogs
        reward = 0
        pos = self.robot_position
        ori = self.robot_orientation
        cont = self.contact_state

        [x_Goal, y_Goal, z_Goal] = [1.0,1.0,0.5]  # x,y position of goal
        Goal_reward = 100  # define Goal reward
        Fall_penalty = -10  # define Fall penalty
        Contact_penalty = -5  # define Contact Point Penalty
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

        # Goal reward
        if px == x_Goal and py == y_Goal:
            reward += Goal_reward
        
        # Fall Penalty
        if pz == 0:
            reward += Fall_penalty
        
        # Contact Point Penalty
        if [L1, L2, R1, R2] == [1,1,1,1] or [L1, L2] == [0,0] or [R1, R2] == [0,0]:
            reward += Contact_penalty

        # Progress reward
        reward += math.exp(-(abs(px-x_Goal) + abs(py-y_Goal)))
        # reward += k2 * math.exp(-(abs(px-x_Goal) + abs(py-y_Goal))) # an additional coefficient can be added

        # Mass Centre reward
        reward += math.exp(-abs(pz-z_Goal))
        # reward += k3 * math.exp(-abs(pz-z_Goal))  # an additional coefficient can be added

        # Stability reward
        reward += min(0, r_allow - abs(r)) + min(0, p_allow - abs(p))
        # reward += k5 * (min(0, r_allow - abs(r)) + min(0, p_allow - abs(p)))  # an additional coefficient can be added

        return reward

    def calculate_rpy(self, Quaternion):  # convert quaternion to roll pitch yaw (r, p, y)
        
        # Quaternion
        qw = Quaternion[0] 
        qx = Quaternion[1]
        qy = Quaternion[2]
        qz = Quaternion[3]

        # get roll pitch yaw in radians
        r = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        #p = math.asin(2 * (qw * qy - qz * qx))
        p = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))  # avoid exceeding range of asin
        y = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        # print([r,p,y])  # for testing only
 
        # get roll pitch yaw in degrees
        r = math.degrees(r)
        p = math.degrees(p)
        y = math.degrees(y)

        return [r, p, y]
    
    def step(self,t) -> None:

        if self.first_step:
            self.robot.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            action = self.generate_alternating_gait(t)
            asyncio.run(self.control_joints_async(action))
            # self._anymal.advance(step_size, action)
        
        # Take a simulation step
        self._world.step(render=True)

        # Get sensor data
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if done
        done = self.is_done()

        return observation, reward, done, {}

    def is_done(self):
        # Define the termination condition
        # For example, check if the robot has fallen or the simulation step limit is reached
        return False

    def reset(self):
        self._world.reset()

    def close(self):
        simulation_app.close()

    def run(self) -> None:
        """
        Step simulation based on rendering downtime

        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
        return

def main():
    # Initialize the environment
    dog = Ji_Dog_Env()
    # dog.reset()
    simulation_app.update()
    dog.setup()
    simulation_app.update()
    dog.reset()
    simulation_app.update()
    dog.run()
    simulation_app.close()
    # Define the total number of steps to run
    num_steps = 5
    t = 0
    dt = 0.1  # Time step

    # Assume we have a random action generator
    for step in range(num_steps):
        # Assume each action space is 4-dimensional, with joint action values between [-1, 1]
        # action = np.random.uniform(low=-1.0, high=1.0, size=(4,))

        # Perform one simulation step, return observation, reward, done status, and additional info
        observation, reward, done, info = dog.step(t)
        
        t += dt

        # Print the information of this simulation step
        print(f"Step: {step}")
        # print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        # Exit if the simulation is done
        if done:
            break

    # Close the environment after the simulation is complete
    dog.close()

# Entry point of the main function
if __name__ == "__main__":
    main()
