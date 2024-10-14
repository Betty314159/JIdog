from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import omni.usd
print('1234')

class JidogEnv:
    def __init__(self):
        self.simulation_app = SimulationApp({"headless": True})
        self._world = World()

        self.setup_scene()
        
        self.action_space_size = 4  
        self.state_size = 12  

    def setup_scene(self):
        self._world.scene.add_default_ground_plane()

        usd_path = "/home/bai/ji_dog_robot.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path="/World/Robot")

        self._robot = self._world.scene.get_object("robot_body")
        self._controller = self._robot.get_articulation_controller()
        
        self._world.reset()

    def reset(self):
        self._world.reset()
        return self.get_observation()

    def step(self, action):

        self._controller.apply_action(ArticulationAction(joint_positions=action))
        self._world.step(render=True)

        state = self.get_observation()
        reward = self.get_reward(state, action)
        done = self.is_done(state)
        
        return state, reward, done

    def get_observation(self):
        print('234')
        joint_positions = self._robot.get_joint_positions()
        joint_velocities = self._robot.get_joint_velocities()
        torso_pos, torso_ori = self._robot.get_world_pose()
        print('222')
        
        observation = np.concatenate([joint_positions, joint_velocities, torso_pos, torso_ori])
        print(observation)
        return observation

    def get_reward(self, state, action):
        torso_velocity = state[6:9]  
        forward_velocity_reward = torso_velocity[0]  
        stability_penalty = -np.abs(torso_velocity[1])  
        action_penalty = -np.sum(np.square(action)) 

        reward = forward_velocity_reward + stability_penalty + action_penalty
        return reward

    def is_done(self, state):
        torso_pos = state[:9]
        if torso_pos[2] < 0.2:  
            return True
        return False

    def render(self):
        pass  
    def close(self):
        simulation_app.close()


def main():
    env = JidogEnv()
    print('111')
    state = env.reset()
    print('222')
    done = False
    while not done:
        action = np.random.uniform(low=-1.0, high=1.0, size=(env.action_space_size,))  # Random action
        state, reward, done = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")

    env.close()

if __name__ == "__main__":
    main()
