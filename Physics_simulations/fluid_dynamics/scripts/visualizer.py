import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

# Load all simulation files
files = sorted(glob.glob("../build/fluid_sim_output_step_*.csv"))
data = [np.loadtxt(file, delimiter=',') for file in files]

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(data[0], cmap='viridis', origin='lower')
colorbar = fig.colorbar(cax, ax=ax, label='Intensity')

# Title and labels
ax.set_title("Fluid Simulation")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Animation function
def update(frame):
    cax.set_array(data[frame])
    ax.set_title(f"Fluid Simulation - Step {frame + 1}")

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data), interval=200)  # Adjust interval for speed

# Show the animation
plt.show()