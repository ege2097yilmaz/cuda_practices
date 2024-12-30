import numpy as np
import matplotlib.pyplot as plt
import glob

files = sorted(glob.glob("../datas/fluid_sim_output_step_*.csv"))

if len(files) < 2:
    raise FileNotFoundError("Insufficient simulation files found (need at least first and last frames).")

first_frame = np.loadtxt(files[0], delimiter=',')
last_frame = np.loadtxt(files[-1], delimiter=',')

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(first_frame, cmap='viridis', origin='lower')
axes[0].set_title("First Frame")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")

axes[1].imshow(last_frame, cmap='viridis', origin='lower')
axes[1].set_title("Last Frame")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

cbar = fig.colorbar(axes[0].images[0], ax=axes, location='right', shrink=0.75)
cbar.set_label('Intensity')

plt.tight_layout()
plt.show()
