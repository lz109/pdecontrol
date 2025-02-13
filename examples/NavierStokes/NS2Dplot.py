        

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pde_control_gym.src import NSReward
from stable_baselines3 import PPO
from stable_baselines3 import SAC


# Define function to extract velocity fields from the simulation
def extract_velocity_fields(env, model, timesteps):
    obs, _ = env.reset()
    velocity_data = {}
    captured_timesteps = set()

    for t in range(max(timesteps) + 1):
        if t in timesteps:  # Ensure the first timestep is stored
            try:
                if len(obs) >= 3:
                    u, v, p = obs[:3]  # Extract first three components
                else:
                    raise ValueError(f"Unexpected observation shape at t={t}: {obs.shape}")

                velocity_data[t] = (u, v)
                captured_timesteps.add(t)  # Keep track of captured timesteps

            except ValueError as e:
                print(f"Observation shape mismatch at t={t}: {obs}")
                raise e

        action, _ = model.predict(obs)  
        action = np.array(action).squeeze()  
        obs, _, done, _, _ = env.step(action)

        if done and len(captured_timesteps) < len(timesteps):
            print(f"Environment terminated early at t={t}, but capturing all timesteps...")
            obs, _ = env.reset()  # Reset environment to continue capturing timesteps

    # Ensure we captured all requested timesteps
    if len(captured_timesteps) < len(timesteps):
        raise RuntimeError(f"Failed to capture all required timesteps: {timesteps}")

    return [velocity_data[t] for t in timesteps]  # Return ordered list


# Function to plot velocity field comparison
def plot_velocity_fields(velocity_data, reference_data, x, y):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    
    methods = ["PPO", "SAC"]
    timesteps = [0, 0.2]
    
    for i, method in enumerate(methods):  # Iterate over PPO and SAC
        for j, t in enumerate(timesteps):  # Iterate over timesteps (t=0 and t=0.2)
            ax = axes[j, i]
            u, v = velocity_data[method][j]  # Get velocity field
            ref_u, ref_v = reference_data[j]  # Get reference velocity field
            ref_u = ref_u[:Nx, :Ny]
            ref_v = ref_v[:Nx, :Ny]

            speed = np.sqrt(u**2 + v**2)  # Compute velocity magnitude

            # Debugging: Ensure x, y, speed have the same shape
            print(f"Plotting {method} at t={t}: x.shape={x.shape}, y.shape={y.shape}, speed.shape={speed.shape}")

            # Ensure correct shapes before plotting
            if x.shape != speed.shape:
                x, y = np.meshgrid(np.linspace(0, X, speed.shape[0]), np.linspace(0, Y, speed.shape[1]))

            # Contour plot for velocity magnitude (background)
            contour = ax.contourf(x, y, speed, cmap="Blues")

            # Quiver plot for velocity vectors (red arrows)
            ax.quiver(x, y, u, v, color="black", alpha=0.8, scale=20)
            ax.quiver(x, y, ref_u, ref_v, color="red", alpha=0.8, scale=20)
            # Set titles and labels
            if j == 0:
                ax.set_title(f"{method}", fontsize=12)
            if i == 0:
                ax.set_ylabel(f"$t = {t}$", fontsize=12)

            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("x")
            ax.set_aspect("equal")

    # Adjust layout and add colorbars
    fig.subplots_adjust(right=0.85, hspace=0.3)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # Colorbar position
    fig.colorbar(contour, cax=cbar_ax, label="Velocity Magnitude")

    plt.savefig("NS.png", dpi=300)
    
# Load reference velocity fields
u_target = np.load('target.npz')['u']
v_target = np.load('target.npz')['v']
reference_data = [(u_target[0], v_target[0]), (u_target[-1], v_target[-1])]  # t=0 and t=0.2

# Create environment
env = gym.make("PDEControlGym-NavierStokes2D", **NS2DParameters)
ppoModelPath = "./logsPPO/rl_model_100000_steps"
sacModelPath = "./logsSAC/rl_model_100000_steps"
ppoModel = PPO.load(ppoModelPath)
sacModel = SAC.load(sacModelPath)
# Run PPO and SAC simulations
timesteps = [0, 200]  # Capture velocity at t=0 and t=0.2
velocity_ppo = extract_velocity_fields(env, ppoModel, timesteps)
velocity_sac = extract_velocity_fields(env, sacModel, timesteps)

# Store results in a dictionary
velocity_data = {"PPO": velocity_ppo, "SAC": velocity_sac}
print(f"Velocity data for PPO: {len(velocity_data['PPO'])}, SAC: {len(velocity_data['SAC'])}")

# Create meshgrid for visualization
Nx, Ny = velocity_ppo[0][0].shape 
x, y = np.meshgrid(np.linspace(0, X, Nx), np.linspace(0, Y, Ny), indexing="ij")


# Plot results
plot_velocity_fields(velocity_data, reference_data, x, y)
