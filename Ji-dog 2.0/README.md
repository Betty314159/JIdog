## Project Overview

This project simulates a quadruped robot's locomotion using the NVIDIA Isaac Sim platform. The goal is to optimise the robot's walking strategies through Proximal Policy Optimisation (PPO) reinforcement learning. Additionally, a manual baseline control (e.g., PID controller) is implemented for performance comparison.

## Features

1. **Environment Setup**:
    
    - `ji_dog_env_create_v2.py` and `ji_dog_env_create_v3.py`: Define the robot's simulation environment, including terrain, model, and interaction logic.
    - Support dynamic terrains (e.g., slopes) and custom reward mechanisms.
2. **Robot Modelling**:
    
    - `ji_dog_model_create.py`: Builds the quadruped robot model, including body, legs, and joints.
3. **Reinforcement Learning**:
    
    - PPO algorithm implemented using PyTorch:
        - `ji_dog_net_v1.py`: Basic implementation.
        - `ji_dog_net_v2.py` and `ji_dog_net_v3.py`: Enhanced versions with reward decomposition and LSTM modules for time-series data.
4. **Baseline Control**:
    
    - Includes simple controllers (e.g., PID controllers) as performance baselines.
5. **Reward Mechanisms**:
    
    - Rewards are decomposed into multiple components, such as distance reward, symmetry reward, mass centre reward, etc., enabling detailed analysis of their contributions to learning.

## Project Structure

- **Folders**:
	- **`Model(including video)`**: Contains `.usd` robot model files and videos of simulation results.
	
	- **`Model_Checkpoints`**: Stores trained PPO model checkpoints for easy resumption of training or evaluation.
	
	- **`Previous_Report`**: Contains previous reports  related to the project.

- **Files**:
    - **`ji_dog_model_create.py`**: Creates the Ji Dog robot model with its body, legs, and joints for articulation. The model is saved as a `.usd` file for use in simulation.
	  
	- **`ji_dog_env_create.py`**: Sets up the simulation environment in Isaac Sim, loads the Ji Dog model, and establishes a reinforcement learning environment. The robot can interact with different surfaces (plane or slope).
	- 
	- **`ji_dog_env_create_v2.py` and `ji_dog_env_create_v3.py`**: Define the reinforcement learning environment with custom action spaces, observation spaces, and reward functions.
	
	- **`ji_dog_net.py`**: Contains the PPO implementation, including the neural network architecture (`ActorCritic` class) and the training algorithm (`PPO` class). This file defines the PPO agent's action and evaluation methods.
	
	- **`ji_dog_net.ipynb`**: A Jupyter Notebook used for testing the neural network structure and training the PPO agent. This notebook provides a more interactive way to experiment with model configurations and training.
	
	- **`ji_dog_baseline.ipynb`**: Used for testing and comparing baseline controller performance.
	
	- **`environment.yml`**: Conda environment file listing all dependencies required for the project. This file allows easy setup of the Python environment.
	
	- **`ji_dog_net_v*.py`**: Different versions of the PPO algorithm. Version `v3` is the latest, featuring reward decomposition and gradient clipping.

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
   Use the `environment.yml` file to create a Conda environment with the required dependencies:
	```bash
	conda env create -f environment.yml
	conda activate ji_dog_env
	```

2. **Generate the Ji Dog Model**:
   ```bash
   python ji_dog_model_create.py
   ```

3. **Set Up Simulation Environment**:
   ```bash
   ./your_isaac_sim_path/python.sh your_path/ji_dog_env_create.py
   ```

4. **Train the PPO Agent**:
   Use `ji_dog_net.ipynb` for testing and training in a notebook environment.

## Usage Instructions

### 1. Install Dependencies

Ensure the following software and libraries are installed:

- **NVIDIA Isaac Sim**: Required for running the simulation.
- **Python Environment**:
    - `torch`
    - `gym`
    - `numpy`

## Run Instructions

### 1. Navigate to the Isaac Sim Directory

Move to the Isaac Sim installation folder. Replace the path with your actual installation path if it's different:

```bash
cd /home/yourpath/.local/share/ov/pkg/isaac-sim-4.2.0
```

### 2. Launch Jupyter Notebook

Start the Jupyter Notebook server provided by Isaac Sim:

```bash
./jupyter_notebook.sh
```

- Use this step to open and run the `.ipynb` files associated with the project.
- The notebooks contain interactive code blocks for visualising and testing different parts of the simulation and training.

### 3. Use TensorBoard for Training Logs

To monitor training progress and visualise logs:

```bash
./python.sh tensorboard --logdir="Ji-dog 2.0/runs/Ji_Dog_Training5"
```

- Replace `"Ji-dog 2.0/runs/Ji_Dog_Training5"` with the specific log directory corresponding to your current training run.
- TensorBoard will provide insights into the reward breakdown, losses, and other metrics during training.
### 4. Notebook Overview

- **Training Notebook**: Use the provided `.ipynb` files to run, debug, and modify PPO training logic.
- **Baseline Notebook**: Explore and compare baseline controllers such as PID to evaluate performance differences.
- **Simulation Visualisation**: Notebooks allow real-time monitoring of the robot's behaviour within the Isaac Sim environment.

## File Version Details

- **`v1` to `v3` versions**:
    - `v1`: Basic PPO reinforcement learning implementation.
    - `v2`: Introduced reward decomposition and an improved neural network structure.
    - `v3`: Further optimised with LSTM integration and gradient clipping for better performance on time-series tasks.

---

## Common Issues

1. **Robot Not Moving**:
    
    - Check if the joints and action ranges are correctly defined.
    - Try adjusting parameters in the reward function.
2. **Dependency Errors**:
    
    - Ensure `torch` and `gym` are installed.
3. **Path Issues**:
    
    - Verify that the model path matches the one defined in the code.
