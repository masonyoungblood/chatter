import numpy as np
from chatter.utils import normalize_spec


def test_normalize_spec_scales_between_zero_and_one():
    # Basic range check on normalization helper
    spec = np.array([[-80.0, -10.0], [-30.0, -60.0]])
    norm = normalize_spec(spec, noise_floor=-60.0)
    assert np.isfinite(norm).all()
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_pad_center_zero_and_downsample_preserve_shape():
    from chatter.utils import pad_center_zero, downsample_spectrogram

    # Padding should grow only the time axis and keep mel bins
    spec = np.ones((4, 5))
    padded = pad_center_zero(spec, target_frames=9)
    assert padded.shape == (4, 9)

    # Downsampling should hit the requested target shape
    resized = downsample_spectrogram(padded, target_shape=(2, 3))
    assert resized.shape == (2, 3)
