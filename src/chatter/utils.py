"""
chatter.utils
=============

Utility functions for spectrogram processing and general helpers.
"""

# Silence annoying warning
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated as an API.*"
)

# Import necessary libraries
import os  # noqa: E402
import sys  # noqa: E402
import contextlib  # noqa: E402
from typing import Any, Generator, Iterator, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from skimage.transform import resize  # noqa: E402


def normalize_spec(
    spec: NDArray[np.floating], noise_floor: float = -60.0
) -> NDArray[np.floating]:
    """
    Normalize a spectrogram to the [0, 1] range by clipping to a noise floor and scaling.

    Parameters
    ----------
    spec : np.ndarray
        Input spectrogram in decibel scale. The array may contain NaN or infinite
        values, which will be replaced with finite values using numpy.nan_to_num.
    noise_floor : float, optional
        Minimum decibel value used as a noise floor. All values below this
        threshold are clipped upward to the noise floor before normalization.
        The default is -60.0.

    Returns
    -------
    np.ndarray
        A spectrogram normalized to the [0, 1] range. If the input is constant
        after clipping, a zero array with the same shape is returned.
    """
    # Replace non-finite values with finite numbers to stabilize downstream operations
    spec = np.nan_to_num(spec, copy=False)
    # Clip values to the specified noise floor to suppress very low-energy regions
    spec = np.maximum(spec, noise_floor)

    # Compute global minimum and maximum after clipping
    min_val = np.min(spec)
    max_val = np.max(spec)

    # If there is dynamic range, scale linearly into the [0, 1] interval
    if max_val > min_val:
        return (spec - min_val) / (max_val - min_val)

    # If the spectrogram is effectively constant, return an array of zeros
    return np.zeros_like(spec)


def pad_center_zero(
    spec: NDArray[np.floating], target_frames: int
) -> NDArray[np.floating]:
    """
    Pad the time axis of a spectrogram to a target length, centering the content.

    Parameters
    ----------
    spec : np.ndarray
        Input spectrogram with shape (n_mels, time_frames).
    target_frames : int
        Desired total number of time frames after padding. This value must be
        greater than or equal to the current number of time frames.

    Returns
    -------
    np.ndarray
        Zero-padded spectrogram with shape (n_mels, target_frames), where the
        original content is centered along the time axis.
    """
    # Compute total padding required along the time axis
    pad_total = target_frames - spec.shape[1]
    # Split padding between left and right so that the content remains centered
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Pad with zeros along the time axis
    return np.pad(
        spec, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0
    )


def downsample_spectrogram(
    spec: NDArray[np.floating], target_shape: Tuple[int, int]
) -> NDArray[np.floating]:
    """
    Downsample a spectrogram to a target shape using smooth anti-aliased interpolation.

    Parameters
    ----------
    spec : np.ndarray
        Input spectrogram with shape (height, width).
    target_shape : tuple of int
        Desired output shape given as (target_height, target_width).

    Returns
    -------
    np.ndarray
        Downsampled spectrogram with shape equal to target_shape. The input
        value range is preserved.
    """
    # Use skimage.resize with anti-aliasing enabled while preserving the original value range
    return resize(spec, target_shape, anti_aliasing=True, preserve_range=True)


def chunker(seq: Sequence[Any], size: int) -> Generator[Sequence[Any], None, None]:
    """
    Generate successive fixed-size chunks from a sequence for batch processing.

    Parameters
    ----------
    seq : Sequence
        Input sequence to be partitioned into chunks.
    size : int
        Maximum size of each chunk. The last chunk may be smaller if the length
        of the sequence is not divisible by the chunk size.

    Yields
    ------
    Sequence
        Successive chunks of the input sequence, each with at most 'size'
        elements.
    """
    # Yield successive slices of length 'size' from the input sequence
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


@contextlib.contextmanager
def suppress_stdout_stderr() -> Iterator[None]:
    """
    Context manager to suppress stdout and stderr output, including
    low-level C / C++ writes that bypass Python's sys.stdout/sys.stderr.

    This is useful for silencing overly verbose third-party libraries
    (e.g., TensorFlow Lite via birdnetlib) during localized operations.
    """
    with open(os.devnull, "w") as fnull:
        # Save original Python-level streams
        old_stdout, old_stderr = sys.stdout, sys.stderr
        # Save original file descriptors for stdout/stderr
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)

        try:
            # Redirect Python and OS-level stdout/stderr to /dev/null
            sys.stdout, sys.stderr = fnull, fnull
            os.dup2(fnull.fileno(), stdout_fd)
            os.dup2(fnull.fileno(), stderr_fd)

            yield
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            # Restore original Python-level streams
            sys.stdout, sys.stderr = old_stdout, old_stderr
