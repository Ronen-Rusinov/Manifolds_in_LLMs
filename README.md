# Manifolds in LLMs

Project for studying activation manifolds in LLMs with clustering, neighborhood analysis, and manifold learning.

## Repository layout

- config/ - YAML configuration files (default and profiles)
- data/ - activation data (downloaded)
- experiments/ - analysis scripts and evaluations
- results/ - outputs from experiments and scripts
- scripts/ - core pipeline steps (data pull, clustering, balltree, training)
- slurm/ - SLURM scripts for batch runs
- src/ - shared utilities and models

## Setup

1. Create a Python environment (3.9+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All scripts read defaults from config/default_config.yaml. Use profiles or override with CLI.

- Quick test profile: config/profile_quick_test.yaml
- Production profile: config/profile_production.yaml

See config/README.md for full details.

## Common workflow

1. Pull data:

```bash
python scripts/pull_data_KEEP.py --config config/profile_production.yaml
```

2. Cluster activations:

```bash
python scripts/minibatch_kmeans_TO_REFACTOR.py --config config/profile_production.yaml
```

3. Build BallTree:

```bash
python experiments/produce_balltree.py 2 --config config/profile_production.yaml
```

4. Obtain neighbors (K is configurable):

```bash
python scripts/obtain_10000_nearest_to_centroids_TO_REFACTOR.py 1 --k 10000 --config config/profile_production.yaml
```

5. Run experiments:

```bash
python experiments/activation_norms.py --config config/profile_production.yaml
python experiments/check_locales.py --config config/profile_production.yaml
```

## Notes

- Most scripts accept a --config argument and additional CLI options for key parameters.
- Outputs are written under results/ with subfolders per script.
- For a full pipeline outline, see pipeline-roadmap.md.
