
## Project Structure

- **Folders**:
	- **`Model`**: Contains the Ji Dog model and any visualizations of the model structure.
	- **`Model_Checkpoints`**: Stores trained PPO model checkpoints for easy resumption of training or evaluation.
	- **`Previous_Report`**: Contains prior reports and documentation related to the project.

- **Files**:
    - **`ji_dog_model_create.py`**: Creates the Ji Dog robot model with its body, legs, and joints for articulation. The model is saved as a `.usd` file for use in simulation.
	  
	- **`ji_dog_env_create.py`**: Sets up the simulation environment in Isaac Sim, loads the Ji Dog model, and establishes a reinforcement learning environment. The robot can interact with different surfaces (plane or slope).
	
	- **`ji_dog_net.py`**: Contains the PPO implementation, including the neural network architecture (`ActorCritic` class) and the training algorithm (`PPO` class). This file defines the PPO agent's action and evaluation methods.
	
	- **`ji_dog_net.ipynb`**: A Jupyter Notebook used for testing the neural network structure and training the PPO agent. This notebook provides a more interactive way to experiment with model configurations and training.
### Prerequisites

following dependencies installed:

- NVIDIA Isaac Sim 4.2.0
- Python 3.10
- `numpy`
- `omni.isaac.core`
- `omni.physx`
- `gym` 
- `torch`
### Start

1. **Generate the Ji Dog Model**:
   ```bash
   python ji_dog_model_create.py
   ```

2. **Set Up Simulation Environment**:
   ```bash
   ./your_isaac_sim_path/python.sh your_path/ji_dog_env_create.py
   ```

3. **Train the PPO Agent**:
   use `ji_dog_net.ipynb` for testing and training in a notebook environment.


---
