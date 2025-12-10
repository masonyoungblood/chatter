import numpy as np
import pytest
from scipy.io import wavfile

from chatter.config import make_config


@pytest.fixture
def tiny_config():
    """
    Minimal, fast config for smoke tests (tiny shapes to keep ops quick).
    """
    return make_config(
        {
            "sr": 22050,
            "n_fft": 256,
            "win_length": 256,
            "hop_length": 128,
            "n_mels": 32,
            "fmin": 200,
            "fmax": 8000,
            "target_shape": (32, 32),
            "batch_size": 2,
            "epochs": 1,
            "latent_dim": 4,
            "ae_type": "convolutional",
        }
    )


@pytest.fixture
def random_audio_file(tmp_path, tiny_config):
    """
    Create a 1-second mono WAV of random noise for testing.
    """
    sr = tiny_config["sr"]
    data = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
    path = tmp_path / "noise.wav"
    wavfile.write(path, sr, data)
    return path
