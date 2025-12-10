"""
chatter.config
==============

Configuration management for the chatter package.
"""

# Silence annoying warning
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated as an API.*"
)

# Import necessary libraries
import warnings  # noqa: E402
from copy import deepcopy  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Default configuration dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    # Spectrogram parameters
    "sr": 44100,
    "n_fft": 2048,
    "win_length": 1024,
    "hop_length": 128,
    "n_mels": 224,
    "fmin": 1000,
    "fmax": 10000,
    "target_shape": (128, 128),
    # Preprocessing parameters
    "high_pass": None,
    "low_pass": None,
    "target_dbfs": -20,
    "threshold": 1,
    "compressor_amount": -20,
    "limiter_amount": -10,
    "static": True,
    "fade_ms": 20,
    "skip_noise": 3.0,
    "use_biodenoising": False,
    "biodenoising_model": "biodenoising16k_dns48",
    "use_noisereduce": True,
    "noise_floor": None,
    # Simple segmentation parameters
    "simple_noise_floor": -60,
    "simple_silence_threshold_db": -40,
    "simple_min_silence_length": 0.001,
    "simple_max_unit_length": 0.4,
    "simple_min_unit_length": 0.03,
    # Pykanto segmentation parameters
    # https://github.com/nilomr/pykanto/blob/93df03b45011af6e32c5c809c6f946f0a486e904/pykanto/parameters.py
    "pykanto_noise_floor": -65,
    "pykanto_top_dB": 65,
    "pykanto_max_dB": -30,
    "pykanto_dB_delta": 5,
    "pykanto_min_silence_length": 0.001,
    "pykanto_max_unit_length": 0.4,
    "pykanto_min_unit_length": 0.03,
    "pykanto_gauss_sigma": 3,
    "pykanto_silence_threshold": 0.2,
    # Autoencoder parameters
    "ae_type": "convolutional",
    "latent_dim": 32,
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-4,
    "beta": 0.5,
    # Other parameters
    "seq_bound": 1.0,
    "lag_size": 3,
    "dark_mode": True,
    "font": "Courier",
    "plot_clip_duration": 5.0,
    "vision_checkpoint": "facebook/dinov2-base",
    "vision_device": None,  # Auto-detected if None (mps > cuda > cpu)
}


def get_default_config() -> Dict[str, Any]:
    """
    Return a deep copy of the default configuration dictionary.

    Returns
    -------
    dict
        A copy of DEFAULT_CONFIG that can be safely modified.
    """
    return deepcopy(DEFAULT_CONFIG)


def make_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a finalized configuration dictionary by merging user overrides
    into the default configuration.

    Parameters
    ----------
    user_config : dict or None, optional
        Dictionary with user-specified overrides. Unknown keys trigger a warning.

    Returns
    -------
    dict
        Finalized configuration dictionary.
    """
    # Start from a fresh copy of the default configuration
    config = get_default_config()

    if user_config is not None:
        # Compare user-supplied keys to known configuration keys
        unknown_keys = set(user_config.keys()) - set(config.keys())

        if unknown_keys:
            # Warn the user about any unknown keys that will be ignored
            warnings.warn(
                f"Unknown config keys: {sorted(unknown_keys)}",
                UserWarning,
            )

        # Apply user overrides on top of the default configuration
        for key, value in user_config.items():
            config[key] = value

    return config


def set_plot_style(config: Any) -> None:
    """
    Configure the matplotlib plotting style based on a configuration object.

    Parameters
    ----------
    config : SimpleNamespace or dict-like
        A configuration-like object with attributes such as 'dark_mode' (bool)
        and 'font' (str). The 'dark_mode' attribute controls whether a dark or
        light theme is used. The 'font' attribute specifies a preferred
        sans-serif font family name, which must be installed on the system
        in order to be applied successfully.

    Returns
    -------
    None
        This function updates matplotlib's global rcParams in place.
    """
    # Select dark or light theme based on the 'dark_mode' flag
    if getattr(config, "dark_mode", False):
        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "axes.facecolor": "black",
                "axes.edgecolor": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "text.color": "white",
                "figure.facecolor": "black",
                "figure.edgecolor": "black",
                "savefig.facecolor": "black",
                "savefig.edgecolor": "black",
            }
        )
    else:
        plt.style.use("default")
        plt.rcParams.update(
            {
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "text.color": "black",
                "figure.facecolor": "white",
                "figure.edgecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
            }
        )

    # Optionally set the font family if a preferred font is provided
    if hasattr(config, "font") and config.font:
        try:
            plt.rcParams.update(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": [config.font],
                }
            )
        except Exception as e:
            print(f"Warning: Could not set font to '{config.font}'. Error: {e}")
            print("Please ensure the font is installed on your system.")
