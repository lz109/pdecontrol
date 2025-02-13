import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from pde_control_gym.src.environments2d.navier_stokes2D import central_difference, laplace
import time 
from tqdm import tqdm
from pde_control_gym.src import NSReward


# THIS EXAMPLE SOLVES THE NavierStokes PROBLEM based on optimization

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
desire_states = np.stack([u_target, v_target], axis=-1) # (NT, Nx, Ny, 2)
NS2DParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "Y": Y,
        "dy":dy,
        "action_dim": 1, 
        "reward_class": NSReward(0.1),
        "normalize": False, 
        "reset_init_condition_func": getInitialCondition,
        "boundary_condition": boundary_condition,
        "U_ref": desire_states, 
        "action_ref": 2.0 * np.ones(1000), 
}

# Make the NavierStokes PDE gym
env = gym.make("PDEControlGym-NavierStokes2D", **NS2DParameters)

# Model-Based Optimization to optimize action 
def apply_boundary(a1, a2):
    a1[:,[-1, 0]] = 0.
    a1[[-1,0],:] = 0.
    a2[:,[-1, 0]] = 0.
    a2[[-1,0],:] = 0.
    return a1, a2

total_reward = 0.
U, V = [], []
T = 199

rewards = []
times = []
for experiment_i in range(1):
    np.random.seed(experiment_i)
    env.reset(seed=400)
    s = time.time()
    for t in tqdm(range(T)):
        obs, reward, done, _ , _ = env.step(np.random.uniform(2,4)) 
        U.append(env.unwrapped.u)
        V.append(env.unwrapped.v)

        total_reward += reward
    print("Total Reward:", total_reward)
    u_target = np.load('target.npz')['u']
    v_target = np.load('target.npz')['v']
    u_ref = [2 for _ in range(T)]
    for ite in range(1):
        Lam1, Lam2 = [], []
        Lam1.append(np.zeros_like(U[0]))
        Lam2.append(np.zeros_like(U[0]))
        pressure = np.zeros_like(U[0])
        for t in tqdm(range(T-1)):
            lam1, lam2 = Lam1[-1], Lam2[-1]
            dl1dx, dl1dy = central_difference(lam1,"x",dx), central_difference(lam1, "y", dy)
            dl2dx, dl2dy = central_difference(lam2,"x", dx), central_difference(lam2, "y", dy) 
            laplace_l1, laplace_l2 = laplace(lam1, dx, dy), laplace(lam2, dx, dy)
            dlam1dt = - 2 * dl1dx * U[-1-t] - dl1dy * V[-1-t] - dl2dx * V[-1-t] - 0.1 * laplace_l1 + (U[-1-t]-u_target[-1-t])
            dlam2dt = - 2 * dl2dy * V[-1-t] - dl1dy * U[-1-t] - dl2dx * U[-1-t] - 0.1 * laplace_l2 + (V[-1-t]-v_target[-1-t])
            lam1 = lam1 - dt * dlam1dt
            lam2 = lam2 - dt * dlam2dt
            lam1, lam2 = apply_boundary(lam1, lam2)
            pressure = env.unwrapped.solve_pressure(lam1, lam2, pressure)  

            lam1 = lam1 - dt * central_difference(pressure, "x", dx)
            lam2 = lam2 - dt * central_difference(pressure, "y", dy)
            lam1, lam2 = apply_boundary(lam1, lam2)
            Lam1.append(lam1)
            Lam2.append(lam2)
        Lam1 = Lam1[::-1]
        actions = []
        for t in tqdm(range(T)):
            dl1dx2 = central_difference(Lam1[t], "y", dy)
            actions.append(u_ref[t] - 0.1/0.1 * sum(dl1dx2[-2, :])*5*dx)
        U, V = [], []
        env.reset(seed=400)
        total_reward = 0.
        for t in tqdm(range(T)):
            obs, reward, done, _ , _ = env.step(actions[t])
            U.append(env.unwrapped.u)
            V.append(env.unwrapped.v)

            total_reward += reward
        plt.plot(actions)
        plt.savefig("NSopti.png", dpi=300)
        np.savez('NS_optmization.npz', U=env.unwrapped.U[:,:,:,0], V=env.unwrapped.U[:,:,:,1], desired_U=np.array(u_target), desired_V=np.array(v_target), actions=actions)

# Load reference velocity fields
u_target = np.load('target.npz')['u']
v_target = np.load('target.npz')['v']
reference_data = [(u_target[0], v_target[0]), (u_target[-1], v_target[-1])]  # t=0 and t=0.2

# Create environment
env = gym.make("PDEControlGym-NavierStokes2D", **NS2DParameters)

import numpy as np
import matplotlib.pyplot as plt

# Load saved optimization results
data = np.load('NS_optmization.npz')

U = data['U']  
V = data['V']  
desired_U = data['desired_U']  
desired_V = data['desired_V']  
actions = data['actions'] 

timesteps = [1, 199]  # First and last time step

# Create figure for velocity field comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

X, Y = U.shape[1], U.shape[2]  
x, y = np.meshgrid(np.linspace(0, 1, X), np.linspace(0, 1, Y), indexing="ij")

for i, t in enumerate(timesteps):
    # Get actual velocity fields
    u_actual = U[t]
    v_actual = V[t]
    
    # Get reference velocity fields
    u_ref = desired_U[t]
    v_ref = desired_V[t]
    
    # Compute velocity magnitudes
    speed_actual = np.sqrt(u_actual**2 + v_actual**2)
    speed_ref = np.sqrt(u_ref**2 + v_ref**2)

    # Plot actual velocity field (left column)
    ax = axes[i, 0]
    ax.contourf(x, y, speed_actual, cmap="Blues")  # Background: velocity magnitude
    ax.quiver(x, y, u_actual, v_actual, color="red", alpha=0.8, scale=20, label="Actual")  # Red arrows
    ax.set_title(f"Actual Velocity at t={t}")
    
    # Plot reference velocity field (right column)
    ax = axes[i, 1]
    ax.contourf(x, y, speed_ref, cmap="Blues")  # Background: velocity magnitude
    ax.quiver(x, y, u_ref, v_ref, color="black", alpha=0.8, scale=20, label="Reference")  # Black arrows
    ax.set_title(f"Reference Velocity at t={t}")

# Set shared axis labels
for ax in axes[:, 0]:
    ax.set_ylabel("Y-axis")
for ax in axes[1, :]:
    ax.set_xlabel("X-axis")

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("velocity_field_comparison.png", dpi=300)

