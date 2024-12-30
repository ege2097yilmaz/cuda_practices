import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = np.loadtxt('../build/fluid_sim_output.csv', delimiter=',')

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(data, cmap='viridis', origin='lower')
plt.colorbar(label='Intensity')
plt.title('Fluid Simulation Output')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()