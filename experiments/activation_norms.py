import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_data
from src.config_manager import load_config, add_config_argument
import argparse
import numpy as np

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(description="Analyze activation norms across different layers")

# Activation analysis parameters
parser.add_argument("--layer_for_activation", type=int, help="Primary layer for activation extraction")
parser.add_argument("--layer_alternative", type=int, help="Alternative layer for activation extraction")

add_config_argument(parser)
args = parser.parse_args()
config = load_config(args.config)

# Override config with CLI arguments
if args.layer_for_activation is not None:
    config.model.layer_for_activation = args.layer_for_activation
if args.layer_alternative is not None:
    config.model.layer_alternative = args.layer_alternative

print("Loading data...")
df = load_data.load_all_parquets(timing=True)

print("Obtaining activations...")
activations_layer_primary = np.array(df[f"activation_layer_{config.model.layer_for_activation}"].tolist(), dtype=np.float32)
activations_layer_alt = np.array(df[f"activation_layer_{config.model.layer_alternative}"].tolist(), dtype=np.float32)

print("Shapes of activations:")
print(f"Layer {config.model.layer_for_activation}: {activations_layer_primary.shape}")
print(f"Layer {config.model.layer_alternative}: {activations_layer_alt.shape}")

print("Calculating norms...")
norms_layer_primary = np.linalg.norm(activations_layer_primary, axis=1)
norms_layer_alt = np.linalg.norm(activations_layer_alt, axis=1)

#remove any infinite or NaN values
norms_layer_primary = norms_layer_primary[np.isfinite(norms_layer_primary)]
norms_layer_alt = norms_layer_alt[np.isfinite(norms_layer_alt)]

#Remove extreme outliers (values above the 99th percentile)
percentile_99 = np.percentile(norms_layer_primary, 99)
norms_layer_primary = norms_layer_primary[norms_layer_primary <= percentile_99]

percentile_99 = np.percentile(norms_layer_alt, 99)
norms_layer_alt = norms_layer_alt[norms_layer_alt <= percentile_99]

print("Calculating statistics...")
print(f"Layer {config.model.layer_for_activation} - Mean: {np.mean(norms_layer_primary):.2f}, Std: {np.std(norms_layer_primary):.2f}")
print(f"Layer {config.model.layer_alternative} - Mean: {np.mean(norms_layer_alt):.2f}, Std: {np.std(norms_layer_alt):.2f}")

print("Graphing the norms...")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.visualization.fig_width_standard, config.visualization.fig_height_compact))

ax1.hist(norms_layer_primary, bins=config.visualization.histogram_bins, alpha=0.7, label=f"Layer {config.model.layer_for_activation}", color='blue')
ax1.set_xlabel("Norm Value")
ax1.set_ylabel("Frequency")
ax1.set_title(f"Distribution of Activation Norms - Layer {config.model.layer_for_activation}")
ax1.legend()

ax2.hist(norms_layer_alt, bins=config.visualization.histogram_bins, alpha=0.7, label=f"Layer {config.model.layer_alternative}", color='orange')
ax2.set_xlabel("Norm Value")
ax2.set_ylabel("Frequency")
ax2.set_title(f"Distribution of Activation Norms - Layer {config.model.layer_alternative}")
ax2.legend()

plt.tight_layout()
plt.savefig("activation_norms_distribution.png")