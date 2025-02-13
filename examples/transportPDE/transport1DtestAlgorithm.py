import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from pde_control_gym.src import TunedReward1D
import pde_control_gym

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
        beta[idx] = 5*math.cos(gamma*math.acos(val))
    return beta

# Kernel function solver for backstepping
def solveKernelFunction(theta):
    kappa = np.zeros(len(theta))
    for i in range(0, len(theta)):
        kernelIntegral = 0
        for j in range(0, i):
            kernelIntegral += (kappa[i-j]*theta[j])*dx
        kappa[i] = kernelIntegral  - theta[i]
    return np.flip(kappa)

# Control convolution solver
def solveControl(kernel, u):
    res = 0
    for i in range(len(u)):
        res += kernel[i]*u[i]
    return res*1e-2

# Set initial condition function here
def getInitialCondition(nx):
    return np.ones(nx)*np.random.uniform(1, 10)

# Returns beta functions passed into PDE environment. Currently gamma is always
# set to 7.35, but this can be modified for further problesms
def getBetaFunction(nx):
    return solveBetaFunction(np.linspace(0, 1, nx), 7.35)

# Timestep and spatial step for PDE Solver
# Run testing cases for 5 seconds
T = 5
dt = 1e-4
dx = 1e-2
X = 1

# Normalize to be set below
hyperbolicParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "reward_class": TunedReward1D(int(round(T/dt)), -1e3, 3e2),
        "normalize":None, 
        "sensing_loc": "full", 
        "control_type": "Dirchilet", 
        "sensing_type": None,
        "sensing_noise_func": lambda state: state,
        "limit_pde_state_size": True,
        "max_state_value": 1e10,
        "max_control_value": 20,
        "reset_init_condition_func": getInitialCondition,
        "reset_recirculation_func": getBetaFunction,
        "control_sample_rate": 0.1,
}

# Parameter varies. For SAC and PPO it is the model itself
# For backstepping it is the beta function
def runSingleEpisode(model, env, parameter):
    terminate = False
    truncate = False

    # Holds the resulting states
    uStorage = []

    # Reset Environment
    obs,__ = env.reset()
    uStorage.append(obs)

    i = 0
    rew = 0
    while not truncate and not terminate:
        # use backstepping controller
        action = model(obs, parameter)
        obs, rewards, terminate, truncate, info = env.step(action)
        uStorage.append(obs)
        rew += rewards 
    u = np.array(uStorage)
    return rew, u

def bcksController(obs, beta):
    kernel = solveKernelFunction(beta)
    return solveControl(kernel, obs)

def RLController(obs, model):
    action, _state = model.predict(obs)
    return action

# Make the hyperbolic PDE gym
# Backstepping does not normalize the control inputs while RL algoriths do
hyperbolicParametersBackstepping = hyperbolicParameters.copy()
hyperbolicParametersBackstepping["normalize"] = False

hyperbolicParametersRL = hyperbolicParameters.copy()
hyperbolicParametersRL["normalize"] = True

envBcks = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersBackstepping)
envRL = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersRL)

# Number of test cases to run
num_instances = 10

# For backstepping controller
spatial = np.linspace(dx, X, int(round(X/dx)))
beta = solveBetaFunction(spatial, 7.35)

# Load RL models. # DUMMY ARGUMENTS NEED TO BE MODIFIED
ppoModelPath = "./logsPPO/rl_model_100000_steps"
sacModelPath = "./logsSAC/rl_model_100000_steps"
ppoModel = PPO.load(ppoModelPath)
sacModel = SAC.load(sacModelPath)

# Run comparisons
# Backstepping
# total_bcks_reward = 0
# for i in range(num_instances):
#     rew, _ = runSingleEpisode(bcksController, envBcks, beta)
#     total_bcks_reward += rew
# print("Backstepping Reward Average:", total_bcks_reward/num_instances)

# PPO
total_ppo_reward = 0
for i in range(num_instances):
    rew, _ = runSingleEpisode(RLController, envRL, ppoModel)
    total_ppo_reward += rew
print("PPO Reward Average:", total_ppo_reward/num_instances)

# SAC
total_sac_reward = 0
for i in range(num_instances):
    rew, _ = runSingleEpisode(RLController, envRL, sacModel)
    total_sac_reward += rew
print("SAC Reward Average:", total_sac_reward/num_instances)



# PLOT EXAMPLE PROBLEMS.

# First Build Same Initial Condition Environments
# Set initial condition function here
def getInitialConditionTen(nx):
    return np.ones(nx)*10

def getInitialConditionOne(nx):
    return np.ones(nx)*1

hyperbolicParametersBacksteppingTen = hyperbolicParametersBackstepping.copy()
hyperbolicParametersBacksteppingTen["reset_init_condition_func"] = getInitialConditionTen

hyperbolicParametersBacksteppingOne = hyperbolicParametersBackstepping.copy()
hyperbolicParametersBacksteppingOne["reset_init_condition_func"] = getInitialConditionOne

hyperbolicParametersRLTen = hyperbolicParametersRL.copy()
hyperbolicParametersRLTen["reset_init_condition_func"] = getInitialConditionTen

hyperbolicParametersRLOne = hyperbolicParametersRL.copy()
hyperbolicParametersRLOne["reset_init_condition_func"] = getInitialConditionOne

# Make environments
envBcksTen = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersBacksteppingTen)
envBcksOne = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersBacksteppingOne)

envRLTen = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersRLTen)
envRLOne = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersRLOne)

rewPPOTen, uPPOTen = runSingleEpisode(RLController, envRLTen, ppoModel)
rewPPOOne, uPPOOne = runSingleEpisode(RLController, envRLOne, ppoModel)

rewSACTen, uSACTen = runSingleEpisode(RLController, envRLTen, sacModel)
rewSACOne, uSACOne = runSingleEpisode(RLController, envRLOne, sacModel)

fig = plt.figure(figsize=set_size(433, 0.99, (2, 2), height_add=1))
subfig = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig.suptitle(r"Example trajectories for $u(0, x)=10$ with PPO and SAC")
subfig.subplots_adjust(left=0.03, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)
X = 1
dx = 1e-2
T = 5
spatial = np.linspace(dx, X, uPPOTen.shape[1])  # Ensure correct number of spatial points

temporal = np.linspace(0, T, len(uPPOOne))
meshx, mesht = np.meshgrid(spatial, temporal)

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
ax[0].set_title("PPO", fontsize=12)
ax[1].set_title("SAC", fontsize=12)
for axes in ax:
    for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
        axis._axinfo['axisline']['linewidth'] = 1
        axis._axinfo['axisline']['color'] = "b"
        axis._axinfo['grid']['linewidth'] = 0.2
        axis._axinfo['grid']['linestyle'] = "--"
        axis._axinfo['grid']['color'] = "#d1d1d1"
        axis.set_pane_color((1,1,1))

ax[0].view_init(10, 35)
ax[0].set_xlabel("x", labelpad=-3)
ax[1].set_xlabel("x", labelpad=-3)
ax[0].set_ylabel("Time", labelpad=-3)
ax[1].set_ylabel("Time", labelpad=-3)
ax[0].set_zlabel(r"$u(x, t)$", rotation=90, labelpad=-7)

ax[0].zaxis.set_rotate_label(False)
ax[0].set_xticks([0, 0.5, 1])
ax[0].tick_params(axis='x', which='major', pad=-3)
ax[1].tick_params(axis='x', which='major', pad=-3)
ax[0].tick_params(axis='y', which='major', pad=-3)
ax[1].tick_params(axis='y', which='major', pad=-3)
ax[0].tick_params(axis='z', which='major', pad=-1)
ax[1].tick_params(axis='z', which='major', pad=-1)

ax[0].plot_surface(meshx, mesht, uPPOTen, edgecolor="black",lw=0.2, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[0].view_init(10, 35)
ax[0].zaxis.set_rotate_label(False)
ax[0].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (uPPOTen.transpose())[-1] 
ax[0].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[1].plot_surface(meshx, mesht, uSACTen, edgecolor="black",lw=0.2, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 35)
ax[1].zaxis.set_rotate_label(False)
ax[1].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (uSACTen.transpose())[-1] 
ax[1].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)


plt.savefig("transport.png", dpi=300)