"""
Chatter: a Python library for applying information theory and AI/ML models to animal communication.

This package provides tools for preprocessing audio files, segmenting them into
syllable-like units, training variational autoencoders on spectrogram representations,
and analyzing the resulting latent feature spaces.

Main Classes
------------
Analyzer
    Audio preprocessing, segmentation, and spectrogram creation.
Trainer
    Variational autoencoder training and feature extraction.
FeatureProcessor
    Post-processing, dimensionality reduction, clustering, and visualization.

Example
-------
>>> from chatter import Analyzer, Trainer, FeatureProcessor, make_config
>>> config = make_config({'sr': 22050, 'fmin': 500, 'fmax': 8000})
>>> analyzer = Analyzer(config)
>>> # ... preprocessing and segmentation
>>> trainer = Trainer(config)
>>> # ... training and feature extraction
>>> processor = FeatureProcessor(df, config)
>>> # ... analysis and visualization
"""

from __future__ import annotations

from pathlib import Path

try:  # Python 3.11+
    import tomllib as _toml_loader
except ModuleNotFoundError:  # pragma: no cover - fallback for older Pythons
    import tomli as _toml_loader  # type: ignore[import]


def _load_version() -> str:
    """
    Load the package version from pyproject.toml.

    This makes the version a single source of truth, so bumping the
    version in pyproject.toml automatically propagates everywhere
    that imports chatter.__version__ (including Sphinx docs).
    """
    try:
        root = Path(__file__).resolve().parents[2]
        pyproject = root / "pyproject.toml"
        data = _toml_loader.loads(pyproject.read_text(encoding="utf8"))
        return str(data["project"]["version"])
    except Exception:
        # Fallback for editable / development environments where parsing fails
        return "0.0.0"


# Package metadata
__version__ = _load_version()
__author__ = "Mason Youngblood"
__email__ = "masonyoungblood@gmail.com"

# Public API
__all__ = [
    "Analyzer",
    "Trainer",
    "FeatureProcessor",
    "get_default_config",
    "make_config",
]

# Force Numba to use the simple workqueue buffer instead of OpenMP
import numba  # noqa: E402

numba.config.THREADING_LAYER = "workqueue"

# Silence annoying warning
import warnings  # noqa: E402

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated as an API.*"
)

# Main user-facing classes and functions
from .analyzer import Analyzer  # noqa: E402
from .trainer import Trainer  # noqa: E402
from .features import FeatureProcessor  # noqa: E402
from .config import get_default_config, make_config  # noqa: E402
