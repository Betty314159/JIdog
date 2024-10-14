# Import necessary modules
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import UsdPhysics, Sdf


import omni.usd
import numpy as np
import carb

# create the class
class Create_Ji_dog:
    def __init__(self):
        self.setup_scene()

    def setup_scene(self):
        self._world = World()

        # add ground
        self._world.scene.add_default_ground_plane()

        # define the size of the dog [length, width, height]
        body_size = np.array([4.0, 2.0, 1.0]) 
        leg_length = 1.0

        # create the body of the dog
        body = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Robot/Body",
                name="robot_body",
                position=np.array([0, 0, 1.0]),
                scale=body_size,
                color=np.array([0.0, 1.0, 0.0])
            )
        )

        # create the legs of the dog
        self.create_leg("FrontLeft", [1.5, 1.0, 0.5], leg_length)
        self.create_leg("FrontRight", [1.5, -1.0, 0.5], leg_length)
        self.create_leg("BackLeft", [-1.5, 1.0, 0.5], leg_length)
        self.create_leg("BackRight", [-1.5, -1.0, 0.5], leg_length)

        # create the joints of the dog
        self.create_joint("FrontLeft", 'z')
        self.create_joint("FrontRight", 'z')
        self.create_joint("BackLeft", 'z')
        self.create_joint("BackRight", 'z')

        # reset the world
        self._world.reset()

        # save the model 
        self.save_model()

        # add body
        self._robot = self._world.scene.get_object("robot_body")
        #self._controller = self.get_articulation_controller()

    def create_leg(self, name, position, length):
        leg = self._world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Robot/Legs/{name}",
                name=f"leg_{name}",
                position=np.array(position),
                scale=np.array([0.2, 0.2, length]),
                color=np.array([1.0, 0.0, 0.0])
            )
        )
    
    def create_joint(self, name, axis):
        joint_path = f"/World/Robot/Joints/{name}_joint"
        joint_prim = UsdPhysics.RevoluteJoint.Define(self._world.stage, Sdf.Path(joint_path))

        joint_prim.CreateBody0Rel().SetTargets([Sdf.Path("/World/Robot/Body")])
        joint_prim.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/Robot/Legs/{name}")])

        # Set joint properties
        joint_prim.CreateAxisAttr(axis)  # Set axis of rotation 
        joint_prim.CreateLowerLimitAttr(-60.0)  
        joint_prim.CreateUpperLimitAttr(60.0)  
        joint_prim.GetPrim().CreateAttribute("physics:damping", Sdf.ValueTypeNames.Float).Set(5.0)  

    def save_model(self):
        stage = omni.usd.get_context().get_stage()
        usd_path = "/home/bai/ji_dog_robot.usd" 
        stage.GetRootLayer().Export(usd_path)
        print(f"Model saved to: {usd_path}")

    def run(self, steps_to_simulate=2000):
        for i in range(steps_to_simulate):
            self._world.step(render=True)

    def close(self):
        simulation_app.close()


def main():
    robot = Create_Ji_dog()
    robot.close()

if __name__ == "__main__":
    main()
