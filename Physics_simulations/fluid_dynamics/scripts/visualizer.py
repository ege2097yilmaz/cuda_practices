import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

# Load all simulation files
files = sorted(glob.glob("../datas/fluid_sim_output_step_*.csv"))

if not files:
    raise FileNotFoundError("No simulation output files found (fluid_sim_output_step_*.csv).")

data = []
for file in files:
    try:
        frame = np.loadtxt(file, delimiter=',')
        data.append(frame)
    except Exception as e:
        print(f"Error loading {file}: {e}")

if not data:
    raise ValueError("No valid simulation data loaded.")

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(data[0], cmap='viridis', origin='lower', interpolation='nearest')
colorbar = fig.colorbar(cax, ax=ax, label='Intensity')

ax.set_title("Fluid Simulation")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Animation function
def update(frame):
    cax.set_array(data[frame])
    ax.set_title(f"Fluid Simulation - Step {frame + 1}/{len(data)}")

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data), interval=200)  # Adjust interval for speed

plt.show()