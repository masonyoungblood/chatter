# Silence annoying warning
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated as an API.*"
)

# Import necessary libraries
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import h5py  # noqa: E402
from tqdm import tqdm  # noqa: E402
from pathlib import Path  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from os import cpu_count  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from pydub import AudioSegment  # noqa: E402

# Import local modules
from .data import (  # noqa: E402
    _extract_species_worker,
    _preprocess_wav_worker,
    _process_file_for_segmentation_worker,
    _process_presegmented_file_worker,
    segment_file,
    segment_file_simple,
    preprocess_audio_data,
    compute_spectrogram,
)
from .utils import chunker  # noqa: E402
from .config import set_plot_style  # noqa: E402

# Constants for audio processing
SILENCE_DETECTION_TOP_DB = 40.0  # dB threshold for detecting silent regions


# Main analyzer class
class Analyzer:
    """
    Main analysis class for audio preprocessing, segmentation, and spectrogram creation.

    This class encapsulates the end-to-end pipeline for preparing audio data
    for autoencoder-based analyses. It provides methods for preprocessing
    audio files, segmenting them into syllable-like units, storing unit
    spectrograms in an HDF5 file, and managing associated metadata.

    Attributes
    ----------
    config : dict
        Configuration dictionary containing all pipeline parameters, including
        audio preprocessing, spectrogram generation, and segmentation settings.
    n_jobs : int
        Number of parallel worker processes used for preprocessing and
        segmentation steps.
    """

    # Initialize analyzer
    def __init__(self, config, n_jobs=-1):
        """
        Initialize the Analyzer with a configuration and optional parallelism.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing all pipeline parameters.
        n_jobs : int, optional
            Number of parallel jobs to run for preprocessing and segmentation.
            If set to -1, all available CPU cores are used. The default is -1.
        """
        # Store configuration
        self.config = dict(config)

        # Set default for skipping initial noise if not provided
        if "skip_noise" not in self.config:
            self.config["skip_noise"] = 3.0

        # Determine number of parallel workers
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        # Print number of cores
        print(f"Using {self.n_jobs} cores for parallel processing")

    # Extract species clips
    def extract_species_clips(
        self,
        input_dir,
        output_dir,
        species,
        confidence_threshold=0.5,
        buffer_seconds=1.0,
        batch_size=None,
    ):
        """
        Recursively find audio files, detect a species, and export clips.

        This method scans an input directory for audio files, runs BirdNET to
        detect a specific species, and saves the resulting audio clips to an
        output directory that mirrors the input's structure. The process is
        executed in parallel across multiple CPU cores.

        Parameters
        ----------
        input_dir : str or Path
            The root directory to search for audio files.
        output_dir : str or Path
            The root directory where the output clips will be saved.
        species : str
            The common name of the target species to detect (e.g., "House Finch").
        confidence_threshold : float, optional
            The minimum confidence level (0-1) for a detection to be included.
            The default is 0.5.
        buffer_seconds : float, optional
            The number of seconds to add to the start and end of each detected
            clip. The default is 1.0.
        batch_size : int, optional
            Number of files to process per batch in each parallel submission.
            If None, a default value of 'n_jobs * 2' is used.
        """
        # Set default batch size
        if batch_size is None:
            batch_size = self.n_jobs * 2

        # Convert paths to pathlib objects
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Find all audio files recursively
        supported_formats = [
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".WAV",
            ".MP3",
            ".FLAC",
            ".OGG",
            ".M4A",
        ]
        files_to_process = []

        for fmt in supported_formats:
            files_to_process.extend(input_dir.rglob(f"*{fmt}"))

        # Print number of found files
        print(f"--- Found {len(files_to_process)} audio files to process ---")

        # Initialize progress bar
        pbar = tqdm(total=len(files_to_process), desc=f"Detecting '{species}'")

        # Process files in parallel batches
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Prepare arguments for the worker function
            tasks = [
                (
                    path,
                    input_dir,
                    output_dir,
                    species,
                    confidence_threshold,
                    buffer_seconds,
                )
                for path in files_to_process
            ]

            # Submit tasks to the executor
            futures = [executor.submit(_extract_species_worker, task) for task in tasks]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"A species extraction task generated an exception: {e}")
                pbar.update(1)

        # Close the progress bar
        pbar.close()
        print(
            f"\n--- Species clip extraction complete. Clips saved to {output_dir} ---"
        )

    # Preprocess directory
    def preprocess_directory(self, input_dir, processed_dir, batch_size=None):
        """
        Preprocess all audio files in a directory and its subdirectories.

        This method performs batch preprocessing of audio files by calling
        '_preprocess_wav_worker' in parallel. All supported audio file formats
        are discovered recursively under 'input_dir', preprocessed, and saved
        as standardized WAV files under 'processed_dir', preserving the directory
        structure.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing raw audio files in various formats.
        processed_dir : str or Path
            Directory in which to save preprocessed WAV files. The directory
            structure mirrors that of 'input_dir'.
        batch_size : int, optional
            Number of files to process per batch in each parallel submission.
            If None, a default value of 'n_jobs * 2' is used. The default is
            None.

        Returns
        -------
        None
            This method writes preprocessed files to disk and prints progress
            information but does not return a value.
        """
        # Set default batch size
        if batch_size is None:
            batch_size = self.n_jobs * 2

        # Convert paths to pathlib objects
        input_dir = Path(input_dir)
        processed_dir = Path(processed_dir)

        # Find all audio files recursively
        supported_formats = [
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".WAV",
            ".MP3",
            ".FLAC",
            ".OGG",
            ".M4A",
        ]
        raw_files = []

        for fmt in supported_formats:
            raw_files.extend(input_dir.rglob(f"*{fmt}"))

        # Print number of found files
        print(f"--- Found {len(raw_files)} audio files to preprocess ---")

        # Initialize progress bar
        pbar = tqdm(total=len(raw_files), desc="Preprocessing audio")

        # Process files in parallel batches
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for file_batch in chunker(raw_files, batch_size):
                futures = []
                for raw_file in file_batch:
                    relative_path = raw_file.relative_to(input_dir)
                    output_path = (processed_dir / relative_path).with_suffix(".wav")

                    futures.append(
                        executor.submit(
                            _preprocess_wav_worker,
                            raw_file,
                            output_path,
                            self.config,
                        ),
                    )

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"A preprocessing task generated an exception: {e}")
                    pbar.update(1)

        # Close the progress bar
        pbar.close()
        print(
            f"\n--- Preprocessing complete. Standardized WAV audio saved to {processed_dir} ---"
        )

    # Demo preprocessing
    def demo_preprocessing(self, input_dir):
        """
        Process a single random file (or segment) from the directory in memory and plot a comparison.

        This function replicates the production preprocessing pipeline logic on a
        slice of audio defined by 'plot_clip_duration'. It prioritizes segments
        with active audio content.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing audio files to process.
        """
        # Apply plot style
        set_plot_style(SimpleNamespace(**self.config))
        input_dir = Path(input_dir)

        # Find all audio files in directory recursively
        supported_formats = [
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".WAV",
            ".MP3",
            ".FLAC",
            ".OGG",
            ".M4A",
        ]
        files = [
            f
            for f in input_dir.rglob("*")
            if f.suffix.lower() in supported_formats and f.is_file()
        ]

        # Check that files were found
        if not files:
            print(f"Error: No audio files found in {input_dir}")
            return

        # Choose a random file
        input_path = np.random.choice(files)
        print(f"--- Demoing preprocessing for: {input_path.name} ---")

        # Determine offset and duration

        # Get clip duration from configuration, defaulting to 10s if missing
        clip_duration_sec = self.config.get("plot_clip_duration", 10.0)

        # Determine file duration and find interesting segments
        # Load with librosa first to find onset and offset activity
        try:
            # Load full audio to scan structure; native sample rate is usually faster
            y_scan, sr_scan = librosa.load(str(input_path), sr=None)
            total_duration = librosa.get_duration(y=y_scan, sr=sr_scan)
        except Exception as e:
            print(f"Error scanning file duration: {e}")
            total_duration = 0

        # Initialize offset
        start_offset = 0.0

        # Check duration
        if total_duration > clip_duration_sec:
            # Split into non-silent intervals
            intervals = librosa.effects.split(y_scan, top_db=SILENCE_DETECTION_TOP_DB)

            if len(intervals) > 0:
                # Pick a random interval
                interval_idx = np.random.choice(len(intervals))
                start_sample = intervals[interval_idx][0]

                # Convert to seconds
                interval_start_sec = start_sample / sr_scan

                # Ensure we do not go past the end
                max_start = total_duration - clip_duration_sec
                # Try to start at the interval but clamp to valid range
                start_offset = min(interval_start_sec, max_start)
                # Ensure offset is non-negative
                start_offset = max(0.0, start_offset)
            else:
                # Fallback if the file is entirely silent
                start_offset = np.random.uniform(
                    0.0, total_duration - clip_duration_sec
                )

        # Print segment information
        print(
            f"   Segment: {start_offset:.2f}s - {start_offset + clip_duration_sec:.2f}s"
        )

        # Processing pipeline

        # Load audio file using pydub
        audio = AudioSegment.from_file(input_path)

        # Apply slicing before processing to save time on heavy operations like noise reduction
        start_ms = int(start_offset * 1000)
        end_ms = int((start_offset + clip_duration_sec) * 1000)

        # Apply safety check for audio length versus calculated slice
        if len(audio) > end_ms:
            audio = audio[start_ms:end_ms]
        else:
            # Handle shorter-than-expected files by taking whatever remains after start_ms
            if start_ms < len(audio):
                audio = audio[start_ms:]
            else:
                # If start_ms exceeds audio length, keep the original clip
                audio = audio

        # Process audio using the shared core logic
        # We only process the slice we are visualizing
        audio_int16 = preprocess_audio_data(audio, self.config)

        # Convert back to float for visualization
        # Audio_int16 is int16, so divide by max int16 to get [-1, 1] float
        y_proc = audio_int16.astype(np.float32) / np.iinfo(np.int16).max

        # Visualization

        # Load original using librosa for visual comparison
        # Use the same offset and duration as the processed clip
        y_orig, sr_orig = librosa.load(
            str(input_path),
            sr=self.config["sr"],
            offset=start_offset,
            duration=clip_duration_sec,
        )

        # Generate spectrograms
        spec_orig = compute_spectrogram(y_orig, sr_orig, self.config)
        spec_proc = compute_spectrogram(y_proc, self.config["sr"], self.config)

        # Plot spectrograms
        fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True, sharey=True)

        # Plot original spectrogram
        librosa.display.specshow(
            spec_orig,
            sr=sr_orig,
            hop_length=self.config["hop_length"],
            x_axis="time",
            y_axis="mel",
            ax=axes[0],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
            cmap="viridis",
        )
        axes[0].set_ylabel("Original", fontsize=12)
        axes[0].set_xlabel("")
        axes[0].set_title(f"{input_path.name} (Offset: {start_offset:.2f}s)")

        # Plot processed spectrogram
        librosa.display.specshow(
            spec_proc,
            sr=self.config["sr"],
            hop_length=self.config["hop_length"],
            x_axis="time",
            y_axis="mel",
            ax=axes[1],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
            cmap="viridis",
        )
        axes[1].set_ylabel("Processed", fontsize=12)

        # Use tight layout
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        plt.show()

    # Segment audio and create spectrograms
    def segment_and_create_spectrograms(
        self,
        processed_dir,
        h5_path,
        csv_path,
        simple=False,
        batch_size=None,
        presegment_csv=None,
    ):
        """
        Segment files and save spectrograms, optionally from a pre-segmented CSV.

        If `presegment_csv` is provided, this method bypasses internal segmentation.
        Instead, it reads the given CSV for 'source_file', 'onset', and 'offset'
        information and generates spectrograms for those specific time slices.

        If `presegment_csv` is None, it scans a directory of preprocessed WAV
        files, segments each file into syllable-like acoustic units using
        internal methods, converts segments into spectrograms, and stores them.

        Parameters
        ----------
        processed_dir : str or Path
            Directory containing preprocessed WAV files.
        h5_path : str or Path
            Path to the output HDF5 file for spectrograms.
        csv_path : str or Path
            Path to the CSV file for unit metadata.
        simple : bool, optional
            If True, use simple amplitude-based segmentation. Default is False.
        batch_size : int, optional
            Number of files to process per batch in parallel. Default is 'n_jobs'.
        presegment_csv : str or Path, optional
            Path to a CSV file with pre-defined segmentations. If provided,
            internal segmentation is skipped. Default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame containing metadata for all units.
        """
        # Set default batch size and prepare paths
        if batch_size is None:
            batch_size = self.n_jobs

        processed_dir, h5_path, csv_path = (
            Path(processed_dir),
            Path(h5_path),
            Path(csv_path),
        )

        # Create directories if needed
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Enforce uint8 for efficient storage
        h5_np_dtype = np.uint8

        # Initialize list for all units and futures
        all_units_data = []
        futures = {}
        total_skipped_segments = 0
        total_attempted_segments = 0
        total_segmented_files = 0

        # Create HDF5 file and set up parallel processing tasks
        with h5py.File(h5_path, "w", libver="latest") as hf:
            # Create dataset
            spec_dataset = hf.create_dataset(
                "spectrograms",
                shape=(0, *self.config["target_shape"]),
                maxshape=(None, *self.config["target_shape"]),
                dtype=h5_np_dtype,
                chunks=(1, *self.config["target_shape"]),
            )

            # Start executor
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # If there is presegmented data
                if presegment_csv:
                    print(f"\n--- Loading pre-segmented data from {presegment_csv} ---")
                    preseg_df = pd.read_csv(presegment_csv)

                    # Check required columns
                    required = ["source_file", "onset", "offset"]
                    if not all(col in preseg_df.columns for col in required):
                        raise ValueError(
                            f"presegment_csv must contain columns: {required}"
                        )

                    # Group by source file
                    grouped = preseg_df.groupby("source_file")
                    pbar = tqdm(
                        total=len(grouped), desc="Generating pre-segmented spectrograms"
                    )

                    # Iterate groups
                    for source_file, group_df in grouped:
                        full_path = processed_dir / source_file
                        total_attempted_segments += len(group_df)

                        if full_path.exists():
                            futures[
                                executor.submit(
                                    _process_presegmented_file_worker,
                                    full_path,
                                    group_df,
                                    self.config,
                                )
                            ] = full_path
                        else:
                            print(
                                f"Warning: Could not find file for pre-segmentation: {full_path}"
                            )
                            pbar.update(1)

                # If there is no presegmented data
                else:
                    processed_files = list(processed_dir.rglob("*.wav"))
                    method = (
                        "simple (amplitude-based)"
                        if simple
                        else "pykanto (image-based)"
                    )
                    print(
                        f"\n--- Found {len(processed_files)} files to segment using {method} method ---"
                    )
                    total_segmented_files = len(processed_files)

                    pbar = tqdm(
                        total=len(processed_files),
                        desc="Segmenting and saving spectrograms",
                    )

                    # Iterate files
                    for pf in processed_files:
                        task = (pf, pf.relative_to(processed_dir), self.config, simple)
                        futures[
                            executor.submit(
                                _process_file_for_segmentation_worker, *task
                            )
                        ] = pf

                # Write output files
                # Initialize statistics for dropped segments
                total_dropped_stats = {
                    "max_length": 0,
                    "min_length": 0,
                    "empty": 0,
                    "no_units": 0,
                }

                for future in as_completed(futures):
                    try:
                        metadata, specs, drops = future.result()

                        # Accumulate drop statistics
                        if drops:
                            for k, v in drops.items():
                                total_dropped_stats[k] = (
                                    total_dropped_stats.get(k, 0) + v
                                )

                        if metadata:
                            # Convert spectrograms to array and enforce scaling
                            arr = np.array(specs, dtype=np.float32)
                            arr = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(
                                np.uint8
                            )

                            # Resize dataset and write data
                            current_size = spec_dataset.shape[0]
                            spec_dataset.resize(current_size + len(arr), axis=0)
                            spec_dataset[current_size:] = arr

                            # Update metadata with HDF5 index
                            for i, meta_item in enumerate(metadata):
                                meta_item["h5_index"] = current_size + i
                            all_units_data.extend(metadata)

                    except Exception as e:
                        print(f"Error processing file {futures[future]}: {e}")
                    pbar.update(1)
                pbar.close()

        # If presegmented, calculate skipped segments
        if presegment_csv:
            total_processed_segments = len(all_units_data)
            total_skipped_segments = total_attempted_segments - total_processed_segments

            if total_skipped_segments > 0:
                skipped_percent = (
                    total_skipped_segments / total_attempted_segments
                ) * 100
                print(
                    f"\nWarning: Skipped {total_skipped_segments} segments ({skipped_percent:.1f}%) total."
                )

                # Report breakdown by reason
                if total_dropped_stats["max_length"] > 0:
                    pct = (
                        total_dropped_stats["max_length"] / total_attempted_segments
                    ) * 100
                    print(
                        f"   - {total_dropped_stats['max_length']} Segments ({pct:.1f}%) exceeded 'simple_max_unit_length' ({self.config.get('simple_max_unit_length')}s)"
                    )

                if total_dropped_stats["min_length"] > 0:
                    pct = (
                        total_dropped_stats["min_length"] / total_attempted_segments
                    ) * 100
                    print(
                        f"   - {total_dropped_stats['min_length']} Segments ({pct:.1f}%) were below 'simple_min_unit_length' ({self.config.get('simple_min_unit_length', 0.0)}s)"
                    )

                if total_dropped_stats["empty"] > 0:
                    pct = (
                        total_dropped_stats["empty"] / total_attempted_segments
                    ) * 100
                    print(
                        f"   - {total_dropped_stats['empty']} Segments ({pct:.1f}%) were empty or invalid"
                    )

                # Check for unexplained drops (for example, errors)
                explained_drops = (
                    total_dropped_stats["max_length"]
                    + total_dropped_stats["min_length"]
                    + total_dropped_stats["empty"]
                )
                unexplained = total_skipped_segments - explained_drops
                if unexplained > 0:
                    pct = (unexplained / total_attempted_segments) * 100
                    print(
                        f"   - {unexplained} Segments ({pct:.1f}%) were dropped due to processing errors or other issues"
                    )
        else:
            # For internally segmented data, summarize files where no units
            # were found. This replaces noisy per-file UserWarnings.
            no_units_files = total_dropped_stats.get("no_units", 0)
            if no_units_files > 0 and total_segmented_files > 0:
                pct = (no_units_files / total_segmented_files) * 100
                print(
                    f"\nWarning: {no_units_files} files ({pct:.1f}%) had no units matching the segmentation criteria."
                )

        # Create and save final dataframe
        unit_df = pd.DataFrame(all_units_data)
        print(
            f"\n--- Data preparation complete. Created records for {len(unit_df)} units ---"
        )

        unit_df.to_csv(csv_path, index=False)
        print(f"Spectrograms saved to {h5_path}")
        print(f"Unit metadata saved to {csv_path}")

        return unit_df

    # Demo segmentation
    def demo_segmentation(self, input_dir, simple=False):
        """
        Segment a random file (or clip) in memory and visualize it.

        This function picks a random audio file, applies the configured segmentation
        algorithm (simple or pykanto) to a specific clip, and plots the results
        matching the visual style of 'plot_detected_units'.

        Parameters
        ----------
        input_dir : str or Path
            Directory containing audio files to process.
        simple : bool, optional
            If True, use simple amplitude-based segmentation. If False, use
            image-based segmentation (pykanto). The default is False.
        """
        # Apply the plot style from the configuration
        set_plot_style(SimpleNamespace(**self.config))
        input_dir = Path(input_dir)

        # Find all audio files in directory recursively
        supported_formats = [
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".WAV",
            ".MP3",
            ".FLAC",
            ".OGG",
            ".M4A",
        ]
        files = [
            f
            for f in input_dir.rglob("*")
            if f.suffix.lower() in supported_formats and f.is_file()
        ]

        # Check that files were found
        if not files:
            print(f"Error: No audio files found in {input_dir}")
            return

        # Choose a random file
        input_path = np.random.choice(files)
        print(f"--- Demoing segmentation for: {input_path.name} ---")

        # Get clip duration from configuration, defaulting to 10s if missing
        clip_duration_sec = self.config.get("plot_clip_duration", 10.0)

        # Scan duration and find an active segment
        try:
            y_scan, sr_scan = librosa.load(str(input_path), sr=None)
            total_duration = librosa.get_duration(y=y_scan, sr=sr_scan)
        except Exception as e:
            print(f"Error scanning file duration: {e}")
            total_duration = 0

        # Initialize offset
        start_offset = 0.0

        # Check duration
        if total_duration > clip_duration_sec:
            intervals = librosa.effects.split(y_scan, top_db=SILENCE_DETECTION_TOP_DB)

            if len(intervals) > 0:
                # Choose a random interval
                interval_idx = np.random.choice(len(intervals))
                start_sample = intervals[interval_idx][0]
                interval_start_sec = start_sample / sr_scan

                # Compute maximum valid start and clamp
                max_start = max(0.0, total_duration - clip_duration_sec)
                start_offset = min(interval_start_sec, max_start)
                start_offset = max(0.0, start_offset)
            else:
                # Use random fallback when no intervals are found
                max_start = max(0.0, total_duration - clip_duration_sec)
                start_offset = np.random.uniform(0.0, max_start)
        else:
            # If file is shorter than clip duration, start from beginning
            start_offset = 0.0

        # Print segment information
        print(
            f"   Segment: {start_offset:.2f}s - {start_offset + clip_duration_sec:.2f}s"
        )

        # Load audio clip
        y, sr = librosa.load(
            str(input_path),
            sr=self.config["sr"],
            offset=start_offset,
            duration=clip_duration_sec,
        )

        # Calculate spectrogram
        spec = compute_spectrogram(y, sr, self.config)

        # Perform segmentation
        if simple:
            segments = segment_file_simple(y, sr, self.config)
            method_str = "Simple"
        else:
            segments = segment_file(spec, self.config)
            method_str = "Pykanto"

        # Create plot matching 'plot_detected_units' style
        fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))

        librosa.display.specshow(
            spec,
            sr=sr,
            hop_length=self.config["hop_length"],
            x_axis="time",
            y_axis="mel",
            ax=ax,
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
            cmap="viridis",
        )
        ax.set_title(f"{input_path.name} ({method_str})\n", fontsize=12)

        # Add rectangles for each detected unit
        if segments.size > 0:
            for start, end in segments:
                onset = start
                offset = end

                if onset < clip_duration_sec and offset > 0.0:
                    rect = patches.Rectangle(
                        (max(0.0, onset), self.config["fmin"]),
                        min(offset, clip_duration_sec) - max(0.0, onset),
                        self.config["fmax"] - self.config["fmin"],
                        linewidth=1.5,
                        edgecolor="none",
                        facecolor="white",
                        alpha=0.2,
                    )
                    ax.add_patch(rect)

        # Use tight layout and show plot
        plt.tight_layout()
        plt.show()

    # Load DataFrame
    def load_df(self, metadata_csv_path):
        """
        Load a DataFrame from a CSV file containing unit metadata.

        Parameters
        ----------
        metadata_csv_path : str or Path
            Path to the CSV file containing unit metadata.

        Returns
        -------
        pd.DataFrame or None
            Loaded DataFrame if the file is found and successfully read.
            Returns None if the file is not found.
        """
        print(f"Attempting to load {metadata_csv_path}...")

        try:
            unit_df = pd.read_csv(metadata_csv_path)
            print(f"--- Successfully loaded {metadata_csv_path} ---")
            return unit_df

        except FileNotFoundError:
            print(f"Error: File not found at {metadata_csv_path}")
            return None
