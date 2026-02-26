"""Configuration management for Manifolds in LLMs project.

This module handles:
- Loading YAML configuration files
- Parsing and validating parameters
- CLI argument overrides
- Providing type-safe access to config values
"""

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


@dataclass
class ModelConfig:
    """Model and dimensionality specifications."""
    latent_dim: int = 12
    latent_dim_2d: int = 2
    latent_dim_3d: int = 3
    latent_dim_4d: int = 4
    layer_for_activation: int = 18
    layer_alternative: int = 6
    balltree_leaf_size: int = 40
    procrustes_test_dim: int = 2304


@dataclass
class TrainingConfig:
    """Training and optimization parameters."""
    epochs: int = 300
    epochs_extended: int = 1000
    pretrain_epochs: int = 200
    learning_rate: float = 1.0e-3
    learning_rate_alt: float = 5.0e-3
    patience: int = 20
    patience_extended: int = 50
    regularization_weight: float = 0.1
    noise_level: float = 0.05
    accumulation_steps: int = 6
    random_seed: int = 42


@dataclass
class ClusteringConfig:
    """Clustering and neighborhood specifications."""
    n_centroids: int = 200
    k_neighbors_3d: int = 5000
    k_neighbors_4d: int = 3000
    k_nearest_large: int = 10000
    kmeans_n_init: int = 3
    kmeans_verbose: int = 10
    kmeans_reassignment_ratio: float = 0.05


@dataclass
class DataConfig:
    """Batch and data specifications."""
    batch_size: int = 200000
    train_fraction: float = 0.5
    val_fraction: float = 0.2
    test_train_split: float = 0.7
    n_samples_synthetic: int = 1000
    n_samples_locality: int = 10
    n_samples_visualization: int = 5000
    n_samples_base: int = 5
    procrustes_n_samples: int = 5000


@dataclass
class DimensionalityConfig:
    """Dimensionality reduction parameters."""
    n_components: int = 12
    n_neighbors: int = 10
    n_components_2d: int = 2
    n_components_3d: int = 3
    n_components_4d: int = 4


@dataclass
class TextConfig:
    """Text processing and context parameters."""
    first_n_words: int = 20
    last_n_words: int = 20
    first_n_tokens_isomap: int = 5
    last_n_tokens_isomap: int = 10


@dataclass
class VisualizationConfig:
    """Visualization and plotting parameters."""
    fig_width_standard: int = 12
    fig_height_standard: int = 6
    fig_width_compact: int = 10
    fig_height_compact: int = 8
    fig_width_large: int = 30
    fig_height_large: int = 30
    histogram_bins: int = 50
    histogram_bins_alt: int = 100
    n_samples_visualization: int = 5000
    dim_2d: int = 2
    dim_3d: int = 3
    dim_4d: int = 4
    visualise_every_n_centroids: int = 5


@dataclass
class NumericalConfig:
    """Numerical constants and thresholds."""
    zero_threshold: float = 1.0e-5
    sentinel_value: int = -1
    pca_max_samples: int = 100


@dataclass
class SyntheticDataConfig:
    """Synthetic data generation parameters."""
    noise_level: float = 0.05
    n_noise_dims: int = 118
    n_data_dims: int = 2
    n_total_dims: int = 120


@dataclass
class LoggingConfig:
    """Logging and I/O parameters."""
    log_interval: int = 1
    verbose_level: int = 10


@dataclass
class AutoencoderConfig:
    """Autoencoder-specific training parameters."""
    training_batch_size: int = 10000  # Full batch training (GPU can handle ~10k samples)
    hidden_dim_ratio: float = 0.5  # hidden_dim = ratio * (input_dim + latent_dim)
    epochs: int = 300
    learning_rate: float = 1.0e-3
    patience: int = 20
    regularization_weight: float = 1.0
    train_fraction: float = 0.7  # 70% for training
    val_fraction: float = 0.15  # 15% for validation (remaining 15% for test)


@dataclass
class Config:
    """Complete configuration object with all subsections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dimensionality: DimensionalityConfig = field(default_factory=DimensionalityConfig)
    text: TextConfig = field(default_factory=TextConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    synthetic_data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                result[key] = value
            else:
                result[key] = asdict(value)
        return result

    def __repr__(self) -> str:
        """Pretty print config."""
        return json.dumps(self.to_dict(), indent=2)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default_config.yaml
                    from the config directory.
    
    Returns:
        Config object with loaded parameters.
    
    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if config_path is None:
        # Try to find default config relative to this file
        config_file = Path(__file__).parent.parent / "config" / "default_config.yaml"
    else:
        config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    return _dict_to_config(config_dict)


def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """Convert dictionary to Config object.
    
    Args:
        config_dict: Dictionary loaded from YAML.
    
    Returns:
        Config object with typed subsections.
    """
    config_dict = config_dict or {}
    
    return Config(
        model=ModelConfig(**config_dict.get("model", {})),
        training=TrainingConfig(**config_dict.get("training", {})),
        clustering=ClusteringConfig(**config_dict.get("clustering", {})),
        data=DataConfig(**config_dict.get("data", {})),
        dimensionality=DimensionalityConfig(**config_dict.get("dimensionality", {})),
        text=TextConfig(**config_dict.get("text", {})),
        visualization=VisualizationConfig(**config_dict.get("visualization", {})),
        numerical=NumericalConfig(**config_dict.get("numerical", {})),
        synthetic_data=SyntheticDataConfig(**config_dict.get("synthetic_data", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
        autoencoder=AutoencoderConfig(**config_dict.get("autoencoder", {})),
    )


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add --config argument to an existing parser.
    
    Use this to add just the config file argument to your script's parser,
    keeping --help clean and focused on script-specific options.
    
    Args:
        parser: An ArgumentParser instance to add the argument to.
    
    Example:
        parser = argparse.ArgumentParser(description="My script")
        add_config_argument(parser)
        parser.add_argument("--my-option", type=int, help="My specific option")
        args = parser.parse_args()
        config = load_config(args.config)
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (uses default_config.yaml if not specified)"
    )


def merge_cli_args(config: Config, args: argparse.Namespace) -> Config:
    """Merge CLI arguments into config, with CLI taking precedence.
    
    Args:
        config: Base Config object from YAML.
        args: Parsed arguments from ArgumentParser.
    
    Returns:
        New Config object with CLI overrides applied.
    """
    config_dict = config.to_dict()
    
    # Get all CLI arguments that were explicitly set (not None)
    cli_overrides = {}
    for key, value in vars(args).items():
        if value is not None and key != "config":
            cli_overrides[key] = value
    
    # Apply overrides to appropriate subsections
    if cli_overrides:
        for key, value in cli_overrides.items():
            # Try to find which subsection contains this key
            found = False
            for section_name, section_config in config_dict.items():
                if isinstance(section_config, dict) and key in section_config:
                    config_dict[section_name][key] = value
                    found = True
                    break
            
            if not found:
                # Try single-level keys (shouldn't happen but good for safety)
                if key in config_dict:
                    config_dict[key] = value
    
    return _dict_to_config(config_dict)


def load_config_with_args(
    description: str = "Script with configurable parameters",
    args: Optional[List[str]] = None
) -> Config:
    """DEPRECATED: This function adds too many arguments to script help pages.
    
    Use this pattern instead:
        
        parser = argparse.ArgumentParser(description="Your description")
        add_config_argument(parser)
        args = parser.parse_args()
        config = load_config(args.config)
    
    This keeps your script's --help focused on script-specific options.
    """
    raise NotImplementedError(
        "load_config_with_args() has been deprecated. "
        "Use add_config_argument() to add --config to your parser instead. "
        "See function docstring for example usage."
    )


# Convenience: module-level config instance (can be overridden by scripts)
DEFAULT_CONFIG: Optional[Config] = None


def get_config() -> Config:
    """Get the current config instance (or load default if not set)."""
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = load_config()
    return DEFAULT_CONFIG


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


if __name__ == "__main__":
    # Demo: load and print config
    parser = argparse.ArgumentParser(description="Config demo")
    add_config_argument(parser)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print("Loaded configuration:")
    print(cfg)
