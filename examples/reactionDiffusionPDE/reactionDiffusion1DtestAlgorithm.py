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
def openLoopController(_, _a):
    return 0
# Make the parabolic PDE gym
# Backstepping does not normalize the control inputs while RL algoriths do
parabolicParametersBackstepping = parabolicParameters.copy()
parabolicParametersBackstepping["normalize"] = False

parabolicParametersRL = parabolicParameters.copy()
parabolicParametersRL["normalize"] = True

envBcks = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersBackstepping)
envRL = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersRL)

# Number of test cases to run
num_instances = 10

# Load RL models. # DUMMY ARGUMENTS NEED TO BE MODIFIED
ppoModelPath = "./logsPPO/rl_model_500000_steps"
sacModelPath = "./logsSAC/rl_model_500000_steps"
ppoModel = PPO.load(ppoModelPath)
sacModel = SAC.load(sacModelPath)

# For backstepping controller
spatial = np.linspace(dx, X, int(round(X/dx)))
beta = solveBetaFunction(spatial, 8)
num_instances = 50
# Backstepping. Controller is slow so this will take some time.
total_bcks_reward = 0
kernel = solveKernelFunction(beta)
for i in range(num_instances):
    rew, _ = runSingleEpisode(bcksController, envBcks, kernel)
    total_bcks_reward += rew
print("Backstepping Reward Average:", total_bcks_reward/num_instances)

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
    return np.ones(nx+1)*10

def getInitialConditionOne(nx):
    return np.ones(nx+1)*1

parabolicParametersBacksteppingTen = parabolicParametersBackstepping.copy()
parabolicParametersBacksteppingTen["reset_init_condition_func"] = getInitialConditionTen

parabolicParametersBacksteppingOne = parabolicParametersBackstepping.copy()
parabolicParametersBacksteppingOne["reset_init_condition_func"] = getInitialConditionOne

parabolicParametersRLTen = parabolicParametersRL.copy()
parabolicParametersRLTen["reset_init_condition_func"] = getInitialConditionTen

parabolicParametersRLOne = parabolicParametersRL.copy()
parabolicParametersRLOne["reset_init_condition_func"] = getInitialConditionOne

# Make environments
envBcksTen = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersBacksteppingTen)
envBcksOne = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersBacksteppingOne)

envRLTen = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersRLTen)
envRLOne = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersRLOne)


rewBcksTen, uBcksTen = runSingleEpisode(bcksController, envBcksTen, kernel)
rewBcksOne, uBcksOne = runSingleEpisode(bcksController, envBcksOne, kernel)

rewPPOTen, uPPOTen = runSingleEpisode(RLController, envRLTen, ppoModel)
rewPPOOne, uPPOOne = runSingleEpisode(RLController, envRLOne, ppoModel)

rewSACTen, uSACTen = runSingleEpisode(RLController, envRLTen, sacModel)
rewSACOne, uSACOne = runSingleEpisode(RLController, envRLOne, sacModel)

rewOpenTen,uOpenTen = runSingleEpisode(openLoopController, envBcksTen, _)
rewOpenTen,uOpenOne = runSingleEpisode(openLoopController, envBcksOne, _)

# PLOT EACH EXAMPLE. PLOTS ARE NOT SCALED THE SAME ON Z SO MAY HAVE TO ADJUST
fig = plt.figure(figsize=set_size(433, 0.99, (2, 3), height_add=1))
subfigs = fig.subfigures(nrows=2, ncols=1, hspace=0)

subfig = subfigs[0]
subfig.suptitle(r"Example trajectories for $u(0, x)=1$ with backstepping, PPO, and SAC")
subfig.subplots_adjust(left=0.03, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)

spatial = np.linspace(0, X, int(round(X/dx))+1)
temporal = np.linspace(0, T, len(uPPOOne))
meshx, mesht = np.meshgrid(spatial, temporal)

ax = subfig.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d", "computed_zorder": False})

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
ax[2].set_xlabel("x", labelpad=-3)
ax[0].set_ylabel("Time", labelpad=-3)
ax[2].set_ylabel("Time", labelpad=-3)
ax[1].set_ylabel("Time", labelpad=-3)
ax[0].set_zlabel(r"$u(x, t)$", rotation=90, labelpad=-7)

ax[0].zaxis.set_rotate_label(False)
ax[0].set_xticks([0, 0.5, 1])
ax[0].tick_params(axis='x', which='major', pad=-3)
ax[1].tick_params(axis='x', which='major', pad=-3)
ax[2].tick_params(axis='x', which='major', pad=-3)
ax[0].tick_params(axis='y', which='major', pad=-3)
ax[1].tick_params(axis='y', which='major', pad=-3)
ax[2].tick_params(axis='y', which='major', pad=-3)
ax[0].tick_params(axis='z', which='major', pad=-1)
ax[1].tick_params(axis='z', which='major', pad=-1)
ax[2].tick_params(axis='z', which='major', pad=-1)

ax[0].plot_surface(meshx, mesht, uBcksOne, edgecolor="black",lw=0.2, rstride=50, cstride=5, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
test = np.ones(len(temporal))
vals = (uBcksOne.transpose())[-1] 
ax[0].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[1].plot_surface(meshx, mesht, uPPOOne, edgecolor="black",lw=0.2, rstride=50, cstride=5, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 35)
ax[1].zaxis.set_rotate_label(False)
ax[1].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (uPPOOne.transpose())[-1] 
ax[1].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[2].plot_surface(meshx, mesht, uSACOne, edgecolor="black",lw=0.2, rstride=50, cstride=5, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[2].view_init(10, 35)
ax[2].zaxis.set_rotate_label(False)
ax[2].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (uSACOne.transpose())[-1] 
ax[2].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
ax[0].set_zticks([-10, 0, 10])
ax[1].set_zticks([-10, 0, 10])
ax[2].set_zticks([-10, 0, 10])


subfig = subfigs[1]
subfig.suptitle(r"Example trajectories for $u(0, x)=10$ with backstepping, PPO, and SAC")
subfig.subplots_adjust(left=0.03, bottom=0.05, right=1, top=0.95, wspace=0, hspace=0)
meshx, mesht = np.meshgrid(spatial, temporal)

ax = subfig.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d", "computed_zorder": False})

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
ax[2].set_xlabel("x", labelpad=-3)
ax[0].set_ylabel("Time", labelpad=-3)
ax[2].set_ylabel("Time", labelpad=-3)
ax[1].set_ylabel("Time", labelpad=-3)
ax[0].set_zlabel(r"$u(x, t)$", rotation=90, labelpad=-7)

ax[0].zaxis.set_rotate_label(False)
ax[0].set_xticks([0, 0.5, 1])
ax[0].tick_params(axis='x', which='major', pad=-3)
ax[1].tick_params(axis='x', which='major', pad=-3)
ax[2].tick_params(axis='x', which='major', pad=-3)
ax[0].tick_params(axis='y', which='major', pad=-3)
ax[1].tick_params(axis='y', which='major', pad=-3)
ax[2].tick_params(axis='y', which='major', pad=-3)
ax[0].tick_params(axis='z', which='major', pad=-1)
ax[1].tick_params(axis='z', which='major', pad=-1)
ax[2].tick_params(axis='z', which='major', pad=-1)

ax[0].plot_surface(meshx, mesht, uBcksTen, edgecolor="black",lw=0.2, rstride=50, cstride=5, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
test = np.ones(len(temporal))
vals = (uBcksTen.transpose())[-1] 
ax[0].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[1].plot_surface(meshx, mesht, uPPOTen, edgecolor="black",lw=0.2, rstride=50, cstride=5, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 35)
ax[1].zaxis.set_rotate_label(False)
ax[1].set_xticks([0, 0.5, 1])
test = np.ones(len(temporal))
vals = (uPPOTen.transpose())[-1] 
ax[1].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
 
ax[2].plot_surface(meshx, mesht, uSACTen, edgecolor="black",lw=0.2, rstride=50, cstride=5, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[2].view_init(10, 35)
ax[2].zaxis.set_rotate_label(False)
ax[2].set_xticks([0, 0.5, 1])
ax[0].set_zticks([-20, 0, 20])
ax[1].set_zticks([-20, 0, 20])

test = np.ones(len(temporal))
vals = (uSACTen.transpose())[-1] 
ax[2].plot(test[1:], temporal[1:], vals[1:], color="red", lw=0.1, antialiased=False, rasterized=False)
ax[2].set_zticks([-20, 0, 20])

plt.savefig("reactionDiffusion.png", dpi=300)