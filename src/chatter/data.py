"""
Core data processing utilities for Chatter.

This module also configures logging/noise behavior for a few noisy
dependencies (e.g., TensorFlow Lite via birdnetlib) so that downstream
pipelines like `extract_species_clips` run without spamming the console.
"""

# Silence annoying warnings from specific third‑party packages
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated as an API.*",
)

# Reduce TensorFlow / TF Lite info logging (e.g., XNNPACK delegate messages)
import os  # noqa: E402
import logging  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide INFO and WARNING logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

# Import necessary libraries
import numpy as np  # noqa: E402
import torch  # noqa: E402
import h5py  # noqa: E402
import librosa  # noqa: E402
import noisereduce  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from pydub import AudioSegment  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from skimage.exposure import equalize_hist  # noqa: E402
from skimage.filters.rank import median  # noqa: E402
from skimage.morphology import dilation, disk, erosion  # noqa: E402
from skimage.util import img_as_ubyte  # noqa: E402
from pykanto.signal.segment import find_units  # noqa: E402
from pykanto.signal.filter import gaussian_blur, kernels, norm  # noqa: E402
from pydub.scipy_effects import high_pass_filter, low_pass_filter  # noqa: E402

# Optional imports with fallbacks
try:
    from birdnetlib import Recording  # noqa: E402
    from birdnetlib.analyzer import Analyzer as BirdNETAnalyzer  # noqa: E402

    BIRDNET_AVAILABLE = True
except ImportError:
    BIRDNET_AVAILABLE = False

    class Recording:  # noqa: E402
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "birdnetlib not available. Install it to use species detection features."
            )

    class BirdNETAnalyzer:  # noqa: E402
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "birdnetlib not available. Install it to use species detection features."
            )


# Optional imports with fallbacks
try:
    from audiocomplib import AudioCompressor, PeakLimiter  # noqa: E402

    AUDIO_COMPRESSION_AVAILABLE = True
except ImportError:
    AUDIO_COMPRESSION_AVAILABLE = False

    class AudioCompressor:  # noqa: E402
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "audiocomplib not available. Install it to use audio compression features."
            )

    class PeakLimiter:  # noqa: E402
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "audiocomplib not available. Install it to use audio limiting features."
            )


try:
    from biodenoising import pretrained  # noqa: E402
    from biodenoising.denoiser.dsp import convert_audio  # noqa: E402

    BIODENOISING_AVAILABLE = True
except ImportError:
    BIODENOISING_AVAILABLE = False

    class pretrained:  # noqa: E402
        @staticmethod
        def get(name):
            raise ImportError(
                "biodenoising not available. Install it to use biodenoising features."
            )

    class convert_audio:  # noqa: E402
        @staticmethod
        def convert(*args, **kwargs):
            raise ImportError(
                "biodenoising not available. Install it to use biodenoising features."
            )


# Import local utility functions
from .utils import (  # noqa: E402
    suppress_stdout_stderr,
    normalize_spec,
    pad_center_zero,
    downsample_spectrogram,
)

# Global cache for biodenoising model to avoid reloading in workers
_BIODENOISING_MODEL_CACHE = {}


# Helper to load biodenoising model
def _get_biodenoising_model(model_name):
    """
    Load and cache the biodenoising model.
    """
    if not BIODENOISING_AVAILABLE:
        raise ImportError(
            "biodenoising not available. Install it to use biodenoising features."
        )

    # check if model is already cached
    if model_name not in _BIODENOISING_MODEL_CACHE:
        # get the model function from pretrained module
        if hasattr(pretrained, model_name):
            model_func = getattr(pretrained, model_name)
            # load model to cpu
            model = model_func().to("cpu")
            model.eval()
            # cache the model
            _BIODENOISING_MODEL_CACHE[model_name] = model
        else:
            raise ValueError(f"Biodenoising model {model_name} not found")

    # return cached model
    return _BIODENOISING_MODEL_CACHE[model_name]


# PyTorch dataset for lazy loading of spectrograms from HDF5 file with worker-safe file handling
class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for lazy loading of spectrograms from an HDF5 file.

    This dataset reads spectrograms on demand from an HDF5 dataset named
    'spectrograms'. It is designed to work correctly with parallel DataLoader
    workers by opening a separate file handle per worker process.

    Attributes
    ----------
    h5_path : str
        Path to the HDF5 file containing the spectrograms.
    indices : list of int
        List of integer indices referring to the entries in the 'spectrograms'
        dataset.
    _h5 : h5py.File or None
        Lazily opened HDF5 file handle. It is not pickled across processes.
    """

    # Initialize dataset
    def __init__(self, h5_path, indices):
        """
        Initialize the spectrogram dataset for lazy HDF5 loading.

        Parameters
        ----------
        h5_path : str or Path
            Path to the HDF5 file containing the 'spectrograms' dataset.
        indices : list of int
            List of dataset indices that this instance will expose through
            __getitem__.
        """
        # Store path and indices for lazy loading
        self.h5_path = str(h5_path)
        self.indices = list(indices)
        self._h5 = None

    # Lazy handle opener
    def _require_handle(self):
        """
        Open the HDF5 file handle lazily for the current worker process.

        Returns
        -------
        h5py.File
            Open HDF5 file handle in read-only mode with SWMR support enabled.
        """
        # Open file lazily per worker process
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._h5

    # Get length
    def __len__(self):
        """
        Return the total number of spectrograms in the dataset.

        Returns
        -------
        int
            Number of spectrograms accessible through this dataset.
        """
        return len(self.indices)

    # Get item by index
    def __getitem__(self, idx):
        """
        Retrieve a single spectrogram from the HDF5 file.

        Parameters
        ----------
        idx : int
            Index within the subset defined by 'indices'.

        Returns
        -------
        torch.Tensor
            Spectrogram tensor with shape (1, height, width). If the underlying
            dataset is stored as uint8, values are scaled to the [0, 1] range;
            otherwise, the data is returned as float32 without rescaling.
        """
        # Open HDF5 file lazily per worker
        hf = self._require_handle()

        # Get actual index from subset list
        data_idx = self.indices[idx]

        # Retrieve spectrogram from dataset
        spec = hf["spectrograms"][data_idx]

        # Convert to tensor and add channel dimension
        if spec.dtype == np.uint8:
            return torch.from_numpy(spec).unsqueeze(0).float() / 255.0
        return torch.from_numpy(spec).unsqueeze(0).float()

    # Prepare for pickling
    def __getstate__(self):
        """
        Prepare the dataset for pickling without an open HDF5 file handle.

        Returns
        -------
        dict
            State dictionary for pickling, with the HDF5 file handle set to
            None to avoid cross-process issues.
        """
        # Ensure HDF5 handle is not pickled across processes
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


# Parallelized helpers


# Worker to extract species clips
def _extract_species_worker(args):
    """
    Detect a target species in a single audio file and export clips.

    This worker function is called by parallel processes. It analyzes an audio
    file using BirdNET, filters for a target species above a confidence
    threshold, combines adjacent detections, adds a buffer, and saves each
    resulting interval as a new WAV file.

    Parameters
    ----------
    args : tuple
        A tuple containing the arguments: (input_path, input_dir, output_dir,
        species, confidence_threshold, buffer_seconds).

    Returns
    -------
    bool
        True if processing completes successfully, False otherwise.
    """
    # Unpack arguments
    input_path, input_dir, output_dir, species, confidence_threshold, buffer_seconds = (
        args
    )

    if not BIRDNET_AVAILABLE:
        print(f"BirdNET not available, skipping species detection for {input_path}")
        return False

    try:
        # Load audio file to get duration and for slicing
        audio = AudioSegment.from_file(input_path)
        total_duration_s = len(audio) / 1000.0

        # Analyze audio file with BirdNET
        detections = []
        with suppress_stdout_stderr():
            analyzer = BirdNETAnalyzer()
            recording = Recording(
                analyzer=analyzer,
                path=str(input_path),
                min_conf=confidence_threshold,
            )
            recording.analyze()
            detections = recording.detections

        # Filter for the target species
        species_intervals = []
        for d in detections:
            if d["common_name"] == species:
                species_intervals.append((d["start_time"], d["end_time"]))

        # Return early if no detections are found
        if not species_intervals:
            return True

        # Sort intervals by start time to prepare for merging
        species_intervals.sort(key=lambda x: x[0])

        # Combine adjacent or overlapping intervals
        combined_intervals = [species_intervals[0]]
        for next_start, next_end in species_intervals[1:]:
            last_start, last_end = combined_intervals[-1]
            if next_start <= last_end:
                combined_intervals[-1] = (last_start, max(last_end, next_end))
            else:
                combined_intervals.append((next_start, next_end))

        # Add buffer to each combined interval
        buffered_intervals = []
        for start_s, end_s in combined_intervals:
            buffered_start = max(0, start_s - buffer_seconds)
            buffered_end = min(total_duration_s, end_s + buffer_seconds)
            buffered_intervals.append((buffered_start, buffered_end))

        # Create the relative output directory
        relative_path = input_path.relative_to(input_dir)
        clip_output_dir = output_dir / relative_path.parent
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        # Export each buffered interval as a new WAV file
        for i, (start_s, end_s) in enumerate(buffered_intervals):
            species_name_safe = species.replace(" ", "_")
            output_filename = f"{input_path.stem}_{species_name_safe}_{i + 1}_{start_s:.2f}s-{end_s:.2f}s.wav"
            output_path = clip_output_dir / output_filename

            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            clip = audio[start_ms:end_ms]

            clip.export(output_path, format="wav")

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


# Core logic for preprocessing a single audio segment
def preprocess_audio_data(audio, config):
    """
    Core preprocessing pipeline for a pydub AudioSegment.

    This function applies the full preprocessing chain: fade in/out, format conversion,
    filtering, amplitude normalization, noise reduction (noisereduce or biodenoising),
    compression, limiting, and final normalization.

    Parameters
    ----------
    audio : pydub.AudioSegment
        Input audio segment.
    config : dict
        Configuration dictionary containing preprocessing parameters.

    Returns
    -------
    np.ndarray
        Processed audio data as a 1D numpy array of type int16.
    """
    # Apply short fade-in and fade-out before any other processing
    fade_ms = config.get("fade_ms", 20)
    audio = audio.fade_in(fade_ms).fade_out(fade_ms)

    # Standardize to mono and resample to target sample rate
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(config["sr"])

    # Apply frequency filters if specified
    if config.get("high_pass") is not None:
        audio = high_pass_filter(audio, config["high_pass"], order=10)
    if config.get("low_pass") is not None:
        audio = low_pass_filter(audio, config["low_pass"], order=10)

    # Normalize amplitude to target dBFS level
    audio = audio.apply_gain(config["target_dbfs"] - audio.dBFS)

    # Convert pydub audio to NumPy array for processing
    arr = np.array(audio.get_array_of_samples())
    max_int = float(2 ** (8 * audio.sample_width - 1) - 1)
    audio_float = arr.astype(np.float32) / max_int

    # Sanitize samples and add dithering to prevent silence issues
    audio_float = np.nan_to_num(audio_float)
    audio_float += 1e-10 * np.random.normal(size=audio_float.shape)

    # Apply biodenoising if enabled
    if config.get("use_biodenoising", False):
        # Load configured model (using cache)
        model_name = config.get("biodenoising_model", "biodenoising16k_dns48")
        model = _get_biodenoising_model(model_name)

        # Prepare audio tensor (add channel dimension)
        wav_tensor = torch.from_numpy(audio_float).unsqueeze(0)

        # Convert audio to model sample rate
        wav_model = convert_audio(
            wav_tensor, config["sr"], model.sample_rate, model.chin
        )

        # Set chunking parameters to avoid memory overload
        chunk_seconds = 10
        chunk_size = int(chunk_seconds * model.sample_rate)
        num_frames = wav_model.shape[-1]
        denoised_parts = []

        # Process chunks using the model on CPU
        with torch.no_grad():
            for i in range(0, num_frames, chunk_size):
                # Extract chunk
                end = min(i + chunk_size, num_frames)
                chunk = wav_model[:, i:end]

                # Process chunk (add batch dimension, process, then move to CPU)
                # Padding is not usually required for these UNet models on typical lengths
                processed_chunk = model(chunk[None])[0]
                denoised_parts.append(processed_chunk.cpu())

        # Stitch chunks back together
        denoised_tensor = torch.cat(denoised_parts, dim=-1)

        # Resample back to original sample rate
        denoised_tensor = convert_audio(
            denoised_tensor, model.sample_rate, config["sr"], 1
        )

        # Convert back to NumPy and remove channel dimension
        audio_float = denoised_tensor.squeeze(0).numpy()

    # Apply noisereduce if enabled
    if config.get("use_noisereduce", True):
        audio_float = noisereduce.reduce_noise(
            y=audio_float,
            sr=config["sr"],
            stationary=config["static"],
            n_jobs=-1,
            n_std_thresh_stationary=config["threshold"],
            thresh_n_mult_nonstationary=config["threshold"],
        )

    # Renormalize to the target dBFS (converting back to AudioSegment and then NumPy)
    # Note: we reconvert to pydub here to leverage apply_gain easily; this maintains the existing flow
    audio_int16 = (audio_float * (2**15 - 1)).astype(np.int16)

    audio_seg = AudioSegment(
        audio_int16.tobytes(), frame_rate=config["sr"], sample_width=2, channels=1
    )

    audio_seg = audio_seg.apply_gain(config["target_dbfs"] - audio_seg.dBFS)
    arr = np.array(audio_seg.get_array_of_samples())

    # Recalculate max int based on 16-bit sample width
    max_int = float(2 ** (8 * audio_seg.sample_width - 1) - 1)
    audio_final = arr.astype(np.float32) / max_int
    audio_final = audio_final.reshape(1, -1)

    # Instantiate the compressor and limiter
    compressor = AudioCompressor(
        threshold=config["target_dbfs"] + config["compressor_amount"],
        ratio=10,
        attack_time_ms=1,
        release_time_ms=100,
        variable_release=True,
    )

    limiter = PeakLimiter(
        threshold=config["target_dbfs"] + config["limiter_amount"],
        attack_time_ms=0.01,
        release_time_ms=1.0,
        variable_release=True,
    )

    # Apply compressor and limiter directly to the NumPy array
    if AUDIO_COMPRESSION_AVAILABLE:
        audio_final = compressor.process(audio_final, sample_rate=config["sr"])
        audio_final = limiter.process(audio_final, sample_rate=config["sr"])
        audio_final = audio_final.squeeze()
    else:
        print(
            "Warning: Audio compression/limiting not available. Install audiocomplib for better audio processing."
        )

    # Normalize the final signal and convert to 16-bit integer
    peak = np.max(np.abs(audio_final))
    if peak > 0:
        audio_final = audio_final / peak

    # Apply optional noise floor logic: silence any audio below the noise floor (post-processing)
    noise_floor = config.get("noise_floor")
    if noise_floor is not None:
        # Calculate amplitude threshold from dB noise floor (assuming normalized -1 to 1)
        # Since signal is peak normalized to 1.0 (0 dBFS), noise floor is relative to that
        amplitude_threshold = 10 ** (noise_floor / 20.0)

        # Zero out samples below threshold (avoid division by zero)
        if amplitude_threshold > 0:
            mask = np.abs(audio_final) >= amplitude_threshold
            audio_final = audio_final * mask

    audio_int16_final = (
        np.clip(audio_final, -1.0, 1.0) * np.iinfo(np.int16).max
    ).astype(np.int16)

    # Return final processed audio
    return audio_int16_final


# Preprocess a single audio file with denoising, filtering, and normalization
def _preprocess_wav_worker(input_path, output_path, config):
    """
    Denoise, standardize, and normalize a single audio file in memory.

    This function is designed to be called from parallel worker processes. It
    loads an audio file, converts it to mono, resamples it to a target sampling
    rate, applies optional high-pass and low-pass filters, normalizes amplitude,
    performs noise reduction, applies dynamic range compression and limiting,
    and saves the result as a 16-bit WAV file.

    Parameters
    ----------
    input_path : Path
        Path to the input audio file in any format supported by pydub.
    output_path : Path
        Path at which to save the processed WAV file.
    config : dict
        Configuration dictionary containing preprocessing parameters. Expected
        keys include 'sr' (int, target sample rate), 'target_dbfs' (float,
        target loudness in dBFS), 'static' (bool for stationary noise reduction),
        'threshold' (float for noise reduction), and optionally 'high_pass'
        and 'low_pass' (cutoff frequencies in Hz).

    Returns
    -------
    bool
        True if preprocessing succeeds without raising an exception, and False
        otherwise.
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio file using pydub to handle multiple formats
        audio = AudioSegment.from_file(input_path)

        # Process audio using the shared core logic
        audio_int16 = preprocess_audio_data(audio, config)

        # Save processed audio as WAV file
        wavfile.write(str(output_path), config["sr"], audio_int16)

        # Return success
        return True

    except Exception as e:
        print(f"Error preprocessing {input_path}: {e}")
        return False


# Compute spectrogram from audio array
def compute_spectrogram(y, sr, config):
    """
    Compute a mel spectrogram in dB scale from an audio time series.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.
    config : dict
        Configuration dictionary containing spectrogram parameters (n_fft, hop_length, etc.).

    Returns
    -------
    np.ndarray
        Mel spectrogram in decibel scale.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config["n_fft"],
        win_length=config["win_length"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


# Segment a mel spectrogram using pykanto image-based method
def segment_file(mel_spectrogram, config):
    """
    Segment a full mel spectrogram using the high-fidelity pykanto workflow.

    This function applies an image-processing pipeline inspired by pykanto to
    identify syllable-like acoustic units in a spectrogram. It uses histogram
    equalization, median filtering, morphological operations, and Gaussian
    blurring, followed by pykanto's 'find_units' to obtain onset and offset
    times.

    Parameters
    ----------
    mel_spectrogram : np.ndarray
        Full mel spectrogram in decibel scale, with shape (n_mels, time_frames).
    config : dict
        Configuration dictionary containing segmentation parameters.

    Returns
    -------
    np.ndarray
        Array of segment boundaries with shape (n_segments, 2), where each row
        contains the onset and offset times (in seconds or frames, depending
        on pykanto configuration). If no units are found or an error occurs,
        an empty array with shape (0,) is returned.
    """
    try:
        # Create a copy of the spectrogram for segmentation-only processing
        spec_for_segmentation = np.copy(mel_spectrogram)

        # Apply the noise floor mask to the copy
        noise_floor = config.get("pykanto_noise_floor", -60.0)
        spec_for_segmentation[spec_for_segmentation < noise_floor] = noise_floor

        # Create SimpleNamespace to mimic pykanto parameters object using separated parameters
        params = SimpleNamespace(
            top_dB=config["pykanto_top_dB"],
            gauss_sigma=config["pykanto_gauss_sigma"],
            sr=config["sr"],
            sample_rate=config["sr"],
            hop_length=config["hop_length"],
            n_fft=config["n_fft"],
            window_length=config["win_length"],
            num_mel_bins=config["n_mels"],
            max_dB=config["pykanto_max_dB"],
            dB_delta=config["pykanto_dB_delta"],
            silence_threshold=config["pykanto_silence_threshold"],
            min_silence_length=config["pykanto_min_silence_length"],
            max_unit_length=config["pykanto_max_unit_length"],
            min_unit_length=config["pykanto_min_unit_length"],
        )

        # Perform pykanto image-processing pipeline on the masked spectrogram
        img_eq = equalize_hist(norm(spec_for_segmentation))
        img_med = median(img_as_ubyte(img_eq), disk(2))
        img_eroded = erosion(img_med, kernels.erosion_kern)
        img_dilated = dilation(img_eroded, kernels.dilation_kern)
        img_dilated = dilation(img_dilated, kernels.erosion_kern)
        img_norm = equalize_hist(img_dilated)

        # Normalize image to decibel range
        img_inv = np.interp(
            img_norm,
            (img_norm.min(), img_norm.max()),
            (-params.top_dB, 0.0),
        )

        # Apply Gaussian blur for smoothing
        img_gauss = gaussian_blur(img_inv.astype(float), params.gauss_sigma)

        # Call pykanto find_units function
        mock_dataset = SimpleNamespace(parameters=params)
        onsets, offsets = find_units(mock_dataset, img_gauss)

        # Check if any units were found
        if onsets is None or offsets is None:
            # No units found; return an empty array and let the caller
            # handle any aggregation or summary of these cases.
            return np.array([])

        # Return segments as array of [onset, offset] pairs
        return np.column_stack((onsets, offsets))

    except Exception as e:
        print(f"Error segmenting audio: {e}")
        return np.array([])


# Segment audio using simple amplitude-based method with librosa
def segment_file_simple(y, sr, config):
    """
    Segment an audio array based on amplitude using frame-wise RMS energy.

    This function computes frame-wise RMS energy using short-time analysis,
    converts it to decibels relative to full scale (dBFS), and identifies
    non-silent intervals as contiguous regions where the RMS level exceeds a
    configurable threshold. The threshold can be specified directly in dBFS
    or as a linear amplitude ratio and is combined with a configurable noise
    floor to produce a single decision boundary. Detected regions separated
    by short silences are merged, and the resulting segments are filtered
    by minimum and optional maximum unit length constraints.

    Parameters
    ----------
    y : np.ndarray
        Audio time series as a one-dimensional NumPy array.
    sr : int
        Sample rate of the audio time series in Hz.
    config : dict
        Configuration dictionary containing segmentation parameters.

    Returns
    -------
    np.ndarray
        Array of segment boundaries with shape (n_segments, 2), where each row
        contains the onset and offset times in seconds. If no valid intervals
        are found or an error occurs, an empty array with shape (0,) is
        returned.
    """
    try:
        hop_length = config["hop_length"]
        n_fft = config["n_fft"]

        # Set noise floor (minimum dBFS to consider)
        noise_floor = config.get("simple_noise_floor", -60.0)

        # Compute frame-wise RMS energy and convert to dBFS
        rms = librosa.feature.rms(
            y=y,
            frame_length=n_fft,
            hop_length=hop_length,
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)

        # Derive detection threshold in dBFS
        threshold_db = float(config["simple_silence_threshold_db"])

        # Use a single effective threshold that respects both noise floor and silence threshold
        effective_threshold_db = max(threshold_db, noise_floor)

        # Derive boolean mask of active (non-silent) frames
        active = rms_db > effective_threshold_db

        # Return empty if everything is silent
        if not np.any(active):
            return np.array([])

        # Find contiguous runs of active frames
        active_indices = np.where(active)[0]
        breaks = np.where(np.diff(active_indices) > 1)[0]

        start_indices = np.concatenate(([0], breaks + 1))
        end_indices = np.concatenate((breaks, [len(active_indices) - 1]))

        # Frame indices are [start_frame, end_frame_exclusive)
        intervals_frames = np.vstack(
            [
                active_indices[start_indices],
                active_indices[end_indices] + 1,
            ]
        ).T

        # Convert frame indices to times in seconds
        intervals_sec = intervals_frames.astype(float) * hop_length / float(sr)

        # Ensure we do not exceed the actual audio duration
        audio_duration = len(y) / float(sr)
        intervals_sec[:, 1] = np.minimum(intervals_sec[:, 1], audio_duration)

        # Merge segments separated by short silences
        min_silence_len_sec = config.get("simple_min_silence_length", 0.1)
        merged_intervals = []

        current_start, current_end = intervals_sec[0]
        for start, end in intervals_sec[1:]:
            silence_duration = start - current_end
            if silence_duration < min_silence_len_sec:
                current_end = end
            else:
                merged_intervals.append([current_start, current_end])
                current_start, current_end = start, end
        merged_intervals.append([current_start, current_end])

        # Filter merged segments by duration constraints
        min_len = config.get("simple_min_unit_length", 0.0)
        max_len = config.get("simple_max_unit_length", float("inf"))

        final_intervals = [
            [start, end]
            for start, end in merged_intervals
            if min_len <= (end - start) <= max_len
        ]

        # Return empty array if no intervals match criteria
        if not final_intervals:
            return np.array([])

        # Return final intervals as NumPy array
        return np.array(final_intervals, dtype=float)

    except Exception as e:
        print(f"Error in simple segmentation: {e}")
        return np.array([])


# Slice full spectrogram into unit spectrograms and process to target shape
def slice_and_process_spectrograms(
    full_spec, segments, config, max_unit_len_sec, min_unit_len_sec=None
):
    """
    Slice and process unit spectrograms from a full spectrogram matrix.

    This function takes a full mel spectrogram and a collection of temporal
    segments, extracts the corresponding time windows, normalizes each segment,
    pads them to a common length, and downsamples them to a fixed target shape.

    Parameters
    ----------
    full_spec : np.ndarray
        Full mel spectrogram with shape (n_mels, time_frames) in decibel scale.
    segments : np.ndarray
        Array of segment boundaries with shape (n_segments, 2), where each row
        contains the onset and offset times in seconds.
    config : dict
        Configuration dictionary containing processing parameters.
    max_unit_len_sec : float
        Maximum unit length in seconds to determine padding.
    min_unit_len_sec : float, optional
        Minimum unit length in seconds. Segments shorter than this are dropped.
        If None, defaults to config['simple_min_unit_length'] or 0.0.

    Returns
    -------
    tuple of (list of np.ndarray, list of int, dict)
        A tuple containing:

        1. List of processed spectrograms, each with shape equal to
           config['target_shape'] and dtype float32.
        2. List of original indices for the returned spectrograms, useful for
           mapping back to metadata if some segments were skipped.
        3. Dictionary containing counts of dropped segments by reason.

        If no segments are provided or all are skipped, empty lists are returned.
    """
    # Initialize list for sliced spectrograms
    sliced_specs = []
    dropped_counts = {"max_length": 0, "min_length": 0, "empty": 0}

    # Convert segment times to spectrogram frame indices
    start_frames = librosa.time_to_frames(
        segments[:, 0],
        sr=config["sr"],
        hop_length=config["hop_length"],
    )
    end_frames = librosa.time_to_frames(
        segments[:, 1],
        sr=config["sr"],
        hop_length=config["hop_length"],
    )

    # Slice full spectrogram for each segment
    for start_frame, end_frame in zip(start_frames, end_frames):
        # Check for invalid slice
        if start_frame >= end_frame:
            sliced_specs.append(None)
            continue

        unit_spec = full_spec[:, start_frame:end_frame]
        normalized = normalize_spec(
            unit_spec, noise_floor=config.get("simple_noise_floor", -60.0)
        )
        sliced_specs.append(normalized)

    # Return empty list if no spectrograms were generated
    if not sliced_specs:
        return [], [], dropped_counts

    # Get maximum unit length in frames
    max_length = librosa.time_to_frames(
        max_unit_len_sec, sr=config["sr"], hop_length=config["hop_length"]
    )

    # Determine minimum unit length (default to 0 if not set)
    if min_unit_len_sec is None:
        min_unit_len_sec = config.get("simple_min_unit_length", 0.0)

    min_length = librosa.time_to_frames(
        min_unit_len_sec, sr=config["sr"], hop_length=config["hop_length"]
    )

    # Pad and downsample each spectrogram to target shape
    final_spectrograms = []
    valid_indices = []

    for i, spec in enumerate(sliced_specs):
        # Check for empty or invalid spectrograms from slicing
        if spec is None:
            dropped_counts["empty"] += 1
            continue

        # Check if segment exceeds maximum length
        if spec.shape[1] > max_length:
            dropped_counts["max_length"] += 1
            continue

        # Check if segment is smaller than minimum length
        if spec.shape[1] < min_length:
            dropped_counts["min_length"] += 1
            continue

        # Check if spectrogram is empty (all zeros)
        if np.max(spec) == 0 and np.min(spec) == 0:
            dropped_counts["empty"] += 1
            continue

        padded = pad_center_zero(spec, max_length)
        final_spec = downsample_spectrogram(padded, config["target_shape"])
        final_spectrograms.append(final_spec.astype(np.float32))
        valid_indices.append(i)

    # Return list of processed spectrograms and valid indices
    return final_spectrograms, valid_indices, dropped_counts


# Process a single file for segmentation and return spectrograms with metadata
def _process_file_for_segmentation_worker(
    processed_file, relative_path, config, simple=False
):
    """
    Load a processed audio file, segment it, and return spectrograms and metadata.

    This helper function is designed to be executed in parallel worker
    processes. It loads a preprocessed audio file, computes a full mel
    spectrogram, performs segmentation using either the pykanto image-based
    method or a simple amplitude-based method, slices the full spectrogram into
    unit spectrograms, and returns both the spectrograms and associated
    metadata for each unit.

    Parameters
    ----------
    processed_file : Path
        Path to the preprocessed audio file (typically a standardized WAV file).
    relative_path : Path
        Relative path from a base directory, used for tracking the source file
        in downstream analyses.
    config : dict
        Configuration dictionary containing spectrogram and segmentation
        parameters.
    simple : bool, optional
        If True, use the simple amplitude-based segmentation method. If False,
        use the pykanto-inspired image-based segmentation on mel spectrograms.
        The default is False.

    Returns
    -------
    tuple of (list, list, dict)
        A tuple (metadata_list, spectrograms_list, dropped_counts).
    """
    # Initialize lists for this file
    file_unit_data = []
    spectrograms_to_return = []
    dropped_counts = {"max_length": 0, "min_length": 0, "empty": 0, "no_units": 0}

    # Load audio file
    try:
        y_full, sr = librosa.load(str(processed_file), sr=config["sr"])
    except Exception as e:
        print(f"Error loading {processed_file}: {e}")
        return file_unit_data, spectrograms_to_return, dropped_counts

    # Initialize audio used for segmentation and time offset
    y_for_segmentation = y_full
    time_offset = 0.0

    # If noise is not static, skip the initial part of the audio for segmentation only
    if not config.get("static", True):
        skip_duration = config.get("skip_noise", 3.0)
        skip_samples = int(skip_duration * sr)
        if len(y_full) > skip_samples:
            y_for_segmentation = y_full[skip_samples:]
            time_offset = skip_duration

    # Compute the full spectrogram from the original, complete audio file
    full_mel_spectrogram_db = compute_spectrogram(y_full, sr, config)

    # Perform segmentation on the (potentially truncated) audio data
    if simple:
        segments_relative = segment_file_simple(y_for_segmentation, sr, config)
        max_unit_len = config["simple_max_unit_length"]
    else:
        # for pykanto, a spectrogram of the truncated audio is needed
        mel_spec_for_segmentation_db = compute_spectrogram(
            y_for_segmentation, sr, config
        )
        segments_relative = segment_file(mel_spec_for_segmentation_db, config)
        max_unit_len = config["pykanto_max_unit_length"]

    # Process segments if any were found
    if segments_relative.size > 0:
        # Adjust segment times to be relative to the start of the original file
        segments_absolute = segments_relative + time_offset

        # Slice from the full, original spectrogram using absolute segment times
        min_unit_len = (
            config["simple_min_unit_length"]
            if simple
            else config.get("pykanto_min_unit_length", config["simple_min_unit_length"])
        )
        spectrograms, valid_indices, dropped_counts = slice_and_process_spectrograms(
            full_mel_spectrogram_db,
            segments_absolute,
            config,
            max_unit_len,
            min_unit_len_sec=min_unit_len,
        )

        # Create metadata using absolute segment times
        for idx, i in enumerate(valid_indices):
            spectrograms_to_return.append(spectrograms[idx])
            file_unit_data.append(
                {
                    "source_file": str(relative_path),
                    "unit_index": i,
                    "onset": float(segments_absolute[i][0]),
                    "offset": float(segments_absolute[i][1]),
                    "max_unit_length_s": float(max_unit_len),
                }
            )
    else:
        # No segments were found for this file (e.g., pykanto reported no units
        # matching the criteria). Record this so the caller can summarize it.
        dropped_counts["no_units"] = 1

    # Return metadata and spectrograms for this file
    return file_unit_data, spectrograms_to_return, dropped_counts


# Process files that have already been segmented
def _process_presegmented_file_worker(processed_file, segments_df, config):
    """
    Generate spectrograms for a single audio file from pre-defined time segments.

    This worker function is called in parallel. It loads a WAV file, computes
    a single mel spectrogram for the entire file, and then slices it into
    individual unit spectrograms based on the onset and offset times provided
    in the `segments_df`. Each slice is then processed to a standard target shape.

    Parameters
    ----------
    processed_file : Path
        The full path to the preprocessed WAV audio file.
    segments_df : pd.DataFrame
        A DataFrame containing the segmentation data for this specific audio
        file, with 'onset' and 'offset' columns in seconds.
    config : dict
        The configuration dictionary containing spectrogram and processing parameters.

    Returns
    -------
    tuple of (list, list, dict)
        A tuple containing a list of metadata dictionaries for each unit,
        a list of the corresponding processed spectrograms as NumPy arrays,
        and a dictionary of dropped segment counts.
    """
    try:
        # Load audio file
        y, sr = librosa.load(str(processed_file), sr=config["sr"])

        # Create a single mel spectrogram for the entire file
        full_mel_spectrogram_db = compute_spectrogram(y, sr, config)

        # Use the existing slice_and_process function, which handles all slicing and processing,
        # defaulting to simple max unit length as a safe default for presegmented data
        segments_array = segments_df[["onset", "offset"]].values
        max_unit_len = config["simple_max_unit_length"]
        spectrograms, valid_indices, dropped_counts = slice_and_process_spectrograms(
            full_mel_spectrogram_db,
            segments_array,
            config,
            max_unit_len,
            min_unit_len_sec=config["simple_min_unit_length"],
        )

        # Create metadata from the DataFrame
        metadata_list = []

        # Filter original DataFrame to only include valid rows
        segments_df_valid = segments_df.iloc[valid_indices]

        for i, (_, row) in enumerate(segments_df_valid.iterrows()):
            metadata_list.append(
                {
                    "source_file": row["source_file"],
                    "unit_index": valid_indices[i],
                    "onset": float(row["onset"]),
                    "offset": float(row["offset"]),
                    "max_unit_length_s": float(max_unit_len),
                }
            )

        return metadata_list, spectrograms, dropped_counts

    except Exception as e:
        print(f"Error processing pre-segmented file {processed_file}: {e}")
        return [], [], {"max_length": 0, "min_length": 0, "empty": 0}
