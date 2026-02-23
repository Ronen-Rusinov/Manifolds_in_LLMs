import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_data
import numpy as np

print("Loading first parquet...")
df = load_data.load_all_parquets(timing=True)

print("Obtaining activations...")
activations_layer_18 = np.array(df[f"activation_layer_18"].tolist(), dtype=np.float16)
activations_layer_6 = np.array(df[f"activation_layer_6"].tolist(), dtype=np.float16)

print("Shapes of activations:")
print(f"Layer 18: {activations_layer_18.shape}")
print(f"Layer 6: {activations_layer_6.shape}")

print("Calculating norms...")
norms_layer_18 = np.linalg.norm(activations_layer_18, axis=1)
norms_layer_6 = np.linalg.norm(activations_layer_6, axis=1)

#remove any infinite or NaN values
norms_layer_18 = norms_layer_18[np.isfinite(norms_layer_18)]
norms_layer_6 = norms_layer_6[np.isfinite(norms_layer_6)]

print("Calculating statistics...")
print(f"Layer 18 - Mean: {np.mean(norms_layer_18):.2f}, Std: {np.std(norms_layer_18):.2f}")
print(f"Layer 6 - Mean: {np.mean(norms_layer_6):.2f}, Std: {np.std(norms_layer_6):.2f}")

print("Graphing the norms...")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(norms_layer_18, bins=50, alpha=0.7, label="Layer 18", color='blue')
ax1.set_xlabel("Norm Value")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Activation Norms - Layer 18")
ax1.legend()

ax2.hist(norms_layer_6, bins=50, alpha=0.7, label="Layer 6", color='orange')
ax2.set_xlabel("Norm Value")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Activation Norms - Layer 6")
ax2.legend()

plt.tight_layout()
plt.savefig("activation_norms_distribution.png")