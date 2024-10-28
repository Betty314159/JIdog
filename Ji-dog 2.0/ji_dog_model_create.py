# ji_dog_model_create.py
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


# Create the class
class Create_Ji_dog:
    def __init__(self):
        self.setup_scene()

    def setup_scene(self):
        self._world = World()

        # Add ground
        self._world.scene.add_default_ground_plane()

        # dDefine the size of the dog [length, width, height]
        body_size = np.array([1.0, 2.0, 0.5])
        leg_length = 0.5

        # Create the body of the dog
        body = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Robot/Body",
                name="robot_body",
                position=np.array([0, 0, 0.5]),
                scale=body_size,
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        # Create the legs of the dog
        self.create_leg("leg1", [0.55, 0.5, 0.25], leg_length)
        self.create_leg("leg2", [-0.55, 0.5, 0.25], leg_length)
        self.create_leg("leg3", [-0.55, -0.5, 0.25], leg_length)
        self.create_leg("leg4", [0.55, -0.5, 0.25], leg_length)

        # Create the joints of the dog
        self.create_joint("joint1", "x", 1)
        self.create_joint("joint2", "x", 2)
        self.create_joint("joint3", "x", 3)
        self.create_joint("joint4", "x", 4)

        # Reset the world
        self._world.reset()

        # Save the model
        self.save_model()

        # Add body
        self._robot = self._world.scene.get_object("robot_body")

    def create_leg(self, name, position, length):
        leg = self._world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Robot/{name}",
                name=f"{name}",
                position=np.array(position),
                scale=np.array([0.2, 0.2, length]),
                color=np.array([1.0, 0.0, 0.0]),
            )
        )

    def create_joint(self, name, axis, idx):
        joint_path = f"/World/Robot/{name}"
        joint_prim = UsdPhysics.RevoluteJoint.Define(
            self._world.stage, Sdf.Path(joint_path)
        )

        joint_prim.CreateBody0Rel().SetTargets([Sdf.Path("/World/Robot/Body")])
        joint_prim.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/Robot/leg{idx}")])

        # Set joint properties
        joint_prim.CreateAxisAttr(axis)  # Set axis of rotation
        joint_prim.CreateLowerLimitAttr(-30.0)
        joint_prim.CreateUpperLimitAttr(30.0)
        joint_prim.GetPrim().CreateAttribute(
            "physics:damping", Sdf.ValueTypeNames.Float
        ).Set(5.0)

    def save_model(self):
        stage = omni.usd.get_context().get_stage()
        usd_path = "/home/bai/ji_dog.usd"
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
