
### Files

- `ji_dog_model_create.py`: Creates the Ji Dog robot model, including the body and legs, and sets up the joints for articulation. The model is saved as a `.usd` file.
- `ji_dog_env_create.py`: Sets up a simulation environment using Isaac Sim, loads the Ji Dog model, and implements a reinforcement learning environment where the robot can move across a plane or slope.

### Prerequisites

following dependencies installed:

- NVIDIA Isaac Sim 4.2.0
- Python 3.10
- `numpy`
- `omni.isaac.core`
- `omni.physx`
- `gym` 

### Example Output

For each step of the simulation, the following information will be printed:

```
Step: 1
Observation: [ ... ]  # Sensor data
Reward: 0.75
Done: False
```

### Simulate the Ji Dog Environment

Run the following command to start the `ji_dog_env_create.py` script:

```bash
./your_isaac_sim_path/python.sh your_path/ji_dog_env_create.py
```
