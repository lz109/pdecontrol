import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import pde_control_gym
from pde_control_gym.src import TunedReward1D


from utils import set_size
from utils import linestyle_tuple
from utils import load_csv

# THIS EXAMPLE TEST A SERIES OF ALGORITHMS AND CALCULATES THE AVERAGE REWARD OF EACH OVER 1K SAMPLES

# NO NOISE
def noiseFunc(state):
    return state

# Chebyshev Polynomial Beta Functions
def solveBetaFunction(x, gamma):
    beta = np.zeros(len(x), dtype=np.float32)
    for idx, val in enumerate(x):
        beta[idx] = 50*math.cos(gamma*math.acos(val))
    return beta

# Kernel function solver for backstepping
def solveKernelFunction(beta):
    k = np.zeros((len(beta), len(beta)))
    # First we calculate a at each timestep
    a = beta

    # FD LOOP

    k[1][1] = -np.sum(a[1] + a[0]) * dx / 4

    for i in range(1, len(beta)-1):
        k[i+1][0] = 0
        k[i+1][i+1] = k[i][i] - dx/4.0 * (np.sum(a[i-1]) + np.sum(a[i]))

        k[i+1][i] = k[i][i] - dx/2 * np.sum(a[i])
        for j in range(1, i):
                k[i+1][j] = -k[i-1][j] + k[i][j+1] + k[i][j-1] + np.sum(a[j])*(dx**2)*(k[i][j+1]+k[i][j-1])/2
    return k

# Control convolution solver
def solveControl(kernel, u):
    return sum(kernel[-1] * u[:len(kernel[-1])]) * dx


# Set initial condition function here
def getInitialCondition(nx):
    return np.ones(nx+1)*np.random.uniform(1, 10)

# Returns beta functions passed into PDE environment. Currently gamma is always
# set to 8, but this can be modified for further problems
def getBetaFunction(nx):
    return solveBetaFunction(np.linspace(0, 1, nx+1), 8)

# Timestep and spatial step for PDE Solver
T = 1
dt = 1e-5
dx = 5e-3
X = 1

# Backstepping does not need to normalize actions to be between -1 and 1, so normalize is set to False. Otherwise,
# parameters are same as RL algorithms
parabolicParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "reward_class": TunedReward1D(int(round(T/dt)), -1e3, 3e2),
        "normalize": None,
        "sensing_loc": "full", 
        "control_type": "Dirchilet", 
        "sensing_type": None,
        "sensing_noise_func": lambda state: state,
        "limit_pde_state_size": True,
        "max_state_value": 1e10,
        "max_control_value": 20,
        "reset_init_condition_func": getInitialCondition,
        "reset_recirculation_func": getBetaFunction,
        "control_sample_rate": 0.001,
}

# Function to run a single episode with sine wave boundary control input
def runSingleEpisode(env, boundary_inputs):
    terminate = False
    truncate = False
    uStorage = []

    # Reset Environment
    obs, _ = env.reset()
    uStorage.append(obs)

    i = 0
    rew = 0
    while not truncate and not terminate and i < len(boundary_inputs):
        action = np.array([boundary_inputs[i]])  # Ensure it matches the expected action shape
        obs, rewards, terminate, truncate, info = env.step(action)
        uStorage.append(obs)
        rew += rewards
        i += 1

    u = np.array(uStorage)
    return rew, u

# Initialize environment
env = gym.make("PDEControlGym-TransportPDE1D", **parabolicParameters)

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the sine wave
frequency = 1  # Frequency in Hz
amplitude = 10  # Amplitude of the sine wave
T = 5  # Total time duration in seconds
num_time_steps = 51  # Match the number of time steps in the solution
interval = T / (num_time_steps - 1)  # Time interval between each saved state

# Generate the sine wave to be used as boundary input
time_points = np.linspace(0, T, num_time_steps)
boundary_inputs = amplitude * np.sin(2 * np.pi * frequency * time_points)

# Run the simulation with the sine wave inputs
reward, solution = runSingleEpisode(env, boundary_inputs)

# Plot the solution at specific time points (1s, 2s, 3s, 4s, 5s)
time_points_to_plot = [1, 2, 3, 4, 5]  # Time points in seconds
indices = [int(tp / interval) for tp in time_points_to_plot]  # Convert time to index

plt.figure(figsize=(10, 6))
for i, idx in enumerate(indices):
    plt.plot(np.linspace(0, X, solution.shape[1]), solution[idx], label=f't = {time_points_to_plot[i]}s')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Solution of the PDE at Specific Time Points')
plt.legend()
plt.grid(True)


plt.savefig("sine.png")
