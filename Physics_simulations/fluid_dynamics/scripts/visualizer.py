import numpy as np
import matplotlib.pyplot as plt
import glob

# Load and visualize all simulation steps
files = sorted(glob.glob("../build/fluid_sim_output_step_*.csv"))

for file in files:
    data = np.loadtxt(file, delimiter=',')
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title(f"Fluid Simulation - {file.split('_')[-1].split('.')[0]} steps")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f"{file.split('.')[0]}.png")  # Save each step as an image
    plt.show()