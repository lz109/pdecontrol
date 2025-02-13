import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pde_control_gym.src import NSReward
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import DQN

# Solving NavierStokes problem using Deep Q Learning

# Set initial condition function here
def getInitialCondition(X):
    u = np.random.uniform(-5, 5) * np.ones_like(X)
    v = np.random.uniform(-5, 5) * np.ones_like(X)
    p = np.random.uniform(-5, 5) * np.ones_like(X)
    return u, v, p

# Set up boundary conditions here
boundary_condition = {
    "upper": ["Controllable", "Dirchilet"],
    "lower": ["Dirchilet", "Dirchilet"],
    "left": ["Dirchilet", "Dirchilet"],
    "right": ["Dirchilet", "Dirchilet"],
}

# Timestep and spatial step for PDE Solver
T = 0.2
dt = 1e-3
dx, dy = 0.05, 0.05
X, Y = 1, 1
u_target = np.load('target.npz')['u']
v_target = np.load('target.npz')['v']
desire_states = np.stack([u_target, v_target], axis=-1)  # (NT, Nx, Ny, 2)
NS2DParameters = {
    "T": T,
    "dt": dt,
    "X": X,
    "dx": dx,
    "Y": Y,
    "dy": dy,
    "action_dim": 1,
    "reward_class": NSReward(0.1),
    "normalize": False,
    "reset_init_condition_func": getInitialCondition,
    "boundary_condition": boundary_condition,
    "U_ref": desire_states,
    "action_ref": 2.0 * np.ones(1000),
}

# Discretize action space 
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins):
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = gym.spaces.Discrete(n_bins)
        self.action_bins = np.linspace(-5, 5, n_bins)

    def action(self, action_index):
        # Convert discrete action index into continuous action
        return np.array([self.action_bins[action_index]])

# Make the NavierStokes PDE gym and wrap it for discrete action space
env = gym.make("PDEControlGym-NavierStokes2D", **NS2DParameters)
discrete_env = DiscreteActionWrapper(env, n_bins=11)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./logsDQN",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Create the DQN model
model = DQN("MlpPolicy", discrete_env, verbose=1, tensorboard_log="./tb_dqn/", device="cpu")

# Train for 200,000 timesteps
model.learn(total_timesteps=2e5, callback=checkpoint_callback)

# Save the final model
model.save("./logsDQN/final_model")
