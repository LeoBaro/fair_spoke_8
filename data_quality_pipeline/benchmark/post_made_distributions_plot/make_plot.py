import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Configurations
max_samples = 5000
data_dirs = ["specificity_scores"]  # Add more directories here later
output_file = "specificity_distribution_plot_max_samples_{max_samples}.png"

def load_specificity_file(filepath):
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]

# Process the first axis (skeleton for all)
raw_file = os.path.join("specificity_scores", "20250620_164249_00000000.txt")
filtered_file = os.path.join("specificity_scores", "full_datacomp_multimodal_specificity_filter_output.txt")

# Load data
raw_img, raw_cap = load_specificity_file(raw_file)
raw_img = raw_img[:max_samples]
raw_cap = raw_cap[:max_samples]
filt_img, filt_cap = load_specificity_file(filtered_file)
filt_img = filt_img[:max_samples]
filt_cap = filt_cap[:max_samples]
    
fig, ax = plt.subplots(1,2, figsize=(8, 6))
density = False
#ax.hist(raw_img, bins=100, alpha=0.5, label="Raw Image")
#ax.hist(filt_img, bins=100, alpha=0.5, label="Filtered Image")
ax[0].hist(raw_cap, bins=50, alpha=0.5, label="Raw Caption", density=density)
ax[0].hist(filt_cap, bins=50, alpha=0.5, label="Filtered Caption", density=density)

# Labeling and legend
ax[0].set_xlabel("Text Specificity", fontsize=12)
ax[0].set_ylabel("Density", fontsize=12)
ax[0].legend(title="Series")

ax[1].hist(raw_img, bins=50, alpha=0.5, label="Raw Image", density=density)
ax[1].hist(filt_img, bins=50, alpha=0.5, label="Filtered Image", density=density)

ax[1].set_xlabel("Image Specificity", fontsize=12)
ax[1].set_ylabel("Density", fontsize=12)
ax[1].legend(title="Series")

fig.savefig(output_file)
plt.tight_layout()
plt.show()

