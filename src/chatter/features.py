# Silence annoying warning
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated as an API.*"
)

# Import necessary libraries
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import h5py  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.cluster import Birch  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Optional imports with fallbacks
try:
    import pacmap  # noqa: E402

    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False

    class pacmap:  # noqa: E402
        class PaCMAP:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "pacmap not available. Install it to use dimensionality reduction features."
                )


try:
    from denmarf import DensityEstimate  # noqa: E402

    DENMARF_AVAILABLE = True
except ImportError:
    DENMARF_AVAILABLE = False

    class DensityEstimate:  # noqa: E402
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "denmarf not available. Install it to use density estimation features."
            )


# Import DTW with fallback for compatibility
try:
    from dtw import dtw  # noqa: E402
except ImportError:
    try:
        from dtw_python import dtw  # noqa: E402
    except ImportError:
        # Fallback implementation if neither package is available
        def dtw(*args, **kwargs):  # noqa: E402
            raise ImportError(
                "DTW package not available. Install 'dtw-python' or 'dtw' package."
            )


# Import local modules
from .config import set_plot_style  # noqa: E402


# Constants for frequency statistics and visualization
FREQ_ENERGY_THRESHOLD = 0.1  # Fraction of max energy to consider "active"
DENSITY_HISTOGRAM_BINS = 500  # Number of bins for 2D density histograms


# Feature processor class
class FeatureProcessor:
    """
    Post-processing class for autoencoder features and associated metadata.

    This class provides methods for dimensionality reduction (PaCMAP),
    clustering (BIRCH), computing within-sequence cosine distances, computing
    VAR-based surprisal scores, assigning sequence identifiers, and visualizing
    embedding structures.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing latent features and associated metadata.
    config : dict
        Configuration dictionary containing post-processing parameters such
        as 'lag_size' and 'seq_bound'.
    """

    # Initialize processor
    def __init__(self, df, config):
        """
        Initialize the FeatureProcessor with a DataFrame and configuration.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing latent features and corresponding metadata.
        config : dict
            Configuration dictionary containing post-processing parameters.
        """
        self.df = df
        self.config = dict(config)

    # Run PaCMAP reduction
    def run_pacmap(self, **kwargs):
        """
        Run PaCMAP dimensionality reduction on latent features and add coordinates.

        This method automatically identifies feature columns, runs PaCMAP to embed them into a two-dimensional space,
        and stores the resulting coordinates in new columns 'pacmap_x' and
        'pacmap_y' in the DataFrame.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed directly to the
            pacmap.PaCMAP constructor, allowing customization of the
            embedding.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining.
        """
        # Automatically infer feature columns
        feature_cols = [col for col in self.df.columns if "_feat_" in col]

        if not feature_cols:
            raise ValueError("No feature columns found in the DataFrame")

        # Extract latent features into a NumPy array
        features = self.df[feature_cols].values

        if not PACMAP_AVAILABLE:
            raise ImportError(
                "pacmap not available. Install it to use dimensionality reduction features."
            )

        # Initialize and fit PaCMAP transformer
        print("--- Running PaCMAP dimensionality reduction ---")
        transformer = pacmap.PaCMAP(**kwargs)
        embedding = transformer.fit_transform(features)

        # Add embedding coordinates to the DataFrame
        self.df["pacmap_x"] = embedding[:, 0]
        self.df["pacmap_y"] = embedding[:, 1]

        print("--- PaCMAP complete ---")
        return self

    # Run BIRCH clustering
    def run_birch_clustering(self, n_clusters_list):
        """
        Run BIRCH clustering for multiple values of 'n_clusters'.

        This method performs BIRCH clustering on the PaCMAP embeddings (columns
        'pacmap_x' and 'pacmap_y') for each requested number of clusters and
        stores cluster labels in separate columns.

        Parameters
        ----------
        n_clusters_list : list of int
            List of 'n_clusters' values for which to compute BIRCH cluster
            assignments. For each value 'n', a column 'birch_n' is added to the
            DataFrame.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining.
        """
        # Automatically infer feature columns
        feature_cols = [col for col in self.df.columns if col.startswith("pacmap_")]

        if not feature_cols:
            raise ValueError(
                "No PaCMAP columns starting with 'pacmap_' found in the DataFrame"
            )

        # Extract features to be used for clustering
        features = self.df[feature_cols].values

        # Iterate through the specified list of cluster numbers
        for n in n_clusters_list:
            print(f"--- Running BIRCH clustering for n_clusters = {n} ---")

            # Initialize and fit BIRCH model
            birch_model = Birch(n_clusters=n)
            labels = birch_model.fit_predict(features)

            # Add cluster labels to a new column in the DataFrame
            self.df[f"birch_{n}"] = labels

        print("--- BIRCH clustering complete ---")
        return self

    # Compute density probability
    def compute_density_probability(self, use_pacmap=False, scaled=True, **kwargs):
        """
        Compute probability density estimates for embeddings using `denmarf`.

        This method fits a Masked AutoRegressive Flow (MAF) density estimator
        to the latent features (or PaCMAP coordinates) and assigns a
        log-probability density score to each unit. High scores indicate
        'typical' points in high-density regions; low scores indicate outlines
        or rare examples.

        Parameters
        ----------
        use_pacmap : bool, optional
            If True, computes density on the 2D 'pacmap_x/y' coordinates
            instead of the full latent space. Default is False
            (recommended: density estimation is more rigorous in the full latent space).
        scaled : bool, optional
            If True, standardizes features (zero mean, unit variance) before
            fitting. Highly recommended for neural density estimators.
            Default is True.
        **kwargs
            Additional keyword arguments to pass to the `denmarf.DensityEstimate`
            constructor or fit method.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining. A new column
            'density_log_prob' is added to the DataFrame.
        """
        if not DENMARF_AVAILABLE:
            raise ImportError(
                "denmarf not available. Install it to use density estimation features."
            )

        print("--- Computing density probability estimates using denmarf ---")

        # Select features
        if use_pacmap:
            if "pacmap_x" not in self.df.columns:
                raise ValueError(
                    "PaCMAP columns not found. Run run_pacmap() first or set use_pacmap=False."
                )
            feature_cols = ["pacmap_x", "pacmap_y"]
            print("    Using 2D PaCMAP coordinates.")
        else:
            feature_cols = [col for col in self.df.columns if "_feat_" in col]
            if not feature_cols:
                raise ValueError("No feature columns found in the DataFrame.")
            print(f"    Using {len(feature_cols)}-dimensional latent feature space.")

        X = self.df[feature_cols].values.astype(np.float32)

        # Scale features
        if scaled:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Determine device for computation (as string for denmarf compatibility)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Initialize density estimator
        de = DensityEstimate(device=device)

        # Capture fit output to avoid cluttering console if verbose
        print(f"    Fitting MAF model on {len(X)} samples (device: {device})...")
        de.fit(X)

        # Score samples
        print("    Scoring samples...")
        log_probs = de.score_samples(X)

        # Save scores to DataFrame
        self.df["density_log_prob"] = log_probs

        print(
            "--- Density estimation complete. Added 'density_log_prob' to DataFrame ---"
        )
        return self

    # Compute cosine distances
    def compute_cosine_distances(self):
        """
        Compute cosine distance between subsequent latent features within sequences.

        This method calculates the cosine distance between each pair of
        consecutive rows that share the same sequence identifier 'seq_id'.
        For each sequence, the first item has an undefined previous neighbor
        and therefore receives a distance of NaN. The results are stored in a
        new column 'cosine_dist'.

        Requirements
        ------------
        The DataFrame must contain:
        - Columns representing latent features.
        - A 'seq_id' column identifying sequences.
        - An 'onset' column to ensure temporal ordering within each sequence.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining.
        """
        print(
            "--- Computing cosine distances between subsequent syllables (within seq_id) ---"
        )

        # Automatically infer feature columns
        feature_cols = [col for col in self.df.columns if "_feat_" in col]
        if not feature_cols:
            raise ValueError("No feature columns found in self.df")
        if "seq_id" not in self.df.columns:
            raise ValueError(
                "compute_cosine_distances requires a 'seq_id' column. "
                "Run assign_sequence_ids() first",
            )

        # Ensure DataFrame is sorted to correctly identify subsequent syllables within sequences
        self.df = self.df.sort_values(by=["seq_id", "onset"]).reset_index(drop=True)

        # Extract features into a NumPy array
        features = self.df[feature_cols].values

        # Compute previous features globally (simple shift by 1)
        prev_features = np.roll(features, shift=1, axis=0)

        # By default, distances will be NaN
        self.df["cosine_dist"] = np.nan

        # Create mask of rows where previous item is within the same sequence
        seq_id_values = self.df["seq_id"].values
        same_seq_mask = seq_id_values == np.roll(seq_id_values, shift=1)

        # Create mask for rows where we actually have a valid previous item within the same sequence
        valid_rows_mask = same_seq_mask

        if np.any(valid_rows_mask):
            # Current and previous features for valid transitions (within same seq_id)
            valid_current_features = features[valid_rows_mask]
            valid_prev_features = prev_features[valid_rows_mask]

            # Compute cosine similarity
            dot = np.einsum("ij,ij->i", valid_current_features, valid_prev_features)
            norm_curr = np.linalg.norm(valid_current_features, axis=1)
            norm_prev = np.linalg.norm(valid_prev_features, axis=1)

            # Avoid division by zero
            denom = norm_curr * norm_prev
            denom = np.where(denom == 0, 1.0, denom)
            similarity = dot / denom

            # Cosine distance is 1 - cosine similarity
            distance = 1.0 - similarity

            # Assign distances to DataFrame
            self.df.loc[valid_rows_mask, "cosine_dist"] = distance

        print("--- Cosine distance calculation complete (within seq_id only) ---")
        return self

    # Compute SSE residuals
    def compute_sse_resid(self):
        """
        Compute VAR-based sum of squared error residuals as a surprisal proxy.

        This method fits a single global vector autoregression (VAR) model with
        a specified lag size across all sequences while respecting sequence
        boundaries defined by 'seq_id'. It then computes per-timestep sum of
        squared error (SSE) residuals for each sequence, including short
        sequences and early time steps using reduced lag orders when necessary.

        Requirements
        ------------
        The DataFrame must contain:
        - Columns representing latent features.
        - A 'seq_id' column identifying sequences.
        Configuration must include:
        - 'lag_size' : int, the lag order p of the VAR model.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining. A new
            column 'sse_resid' is added to the DataFrame containing SSE values
            or NaN where predictions are not defined (for example, the first
            time step of each sequence).
        """
        # Automatically infer feature columns
        feature_cols = [col for col in self.df.columns if "_feat_" in col]
        if not feature_cols:
            raise ValueError("No feature columns found in self.df")
        if "seq_id" not in self.df.columns:
            raise ValueError("compute_sse_resid requires a 'seq_id' column in self.df")

        # Extract latent features and sequence identifiers
        Z_all = self.df[feature_cols].values
        seq_ids = self.df["seq_id"].values

        # Compute basic shapes
        N, K = Z_all.shape
        p = int(self.config["lag_size"])
        if p < 1:
            raise ValueError("config['lag_size'] must be >= 1")

        print(f"--- Running global VAR model (OLS) with lag size p = {p} ---")

        # Find sequence boundaries where seq_id changes
        boundaries = np.where(seq_ids[1:] != seq_ids[:-1])[0] + 1
        seq_starts = np.concatenate(([0], boundaries))
        seq_ends = np.concatenate((boundaries, [N]))

        # Build global design matrix X and target Y from all sequences
        X_rows = []
        Y_rows = []

        for start, end in zip(seq_starts, seq_ends):
            Z_seq = Z_all[start:end]
            T_m = Z_seq.shape[0]

            if T_m <= p:
                # Sequence too short to contribute to global VAR OLS fit
                continue

            # For this sequence, valid targets are t = p..T_m-1 inclusive
            for t in range(p, T_m):
                # Collect lags z_{t-1}, ..., z_{t-p}
                lags = [Z_seq[t - i] for i in range(1, p + 1)]
                X_rows.append(np.concatenate(lags, axis=0))
                Y_rows.append(Z_seq[t])

        if not X_rows:
            raise ValueError(
                "No sequences are longer than lag_size; cannot fit VAR. "
                "Reduce lag_size or provide longer sequences",
            )

        X = np.vstack(X_rows)
        Y = np.vstack(Y_rows)

        # Fit global VAR(p) via multivariate OLS: Y = X * B + intercept
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        Beta, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)

        intercept = Beta[0]
        A_flat = Beta[1:]
        A_mats = []
        for i in range(p):
            A_i = A_flat[i * K : (i + 1) * K].T
            A_mats.append(A_i)

        # Compute per-timestep SSE residuals for each sequence separately
        sse_resid = np.full(N, np.nan, dtype=float)

        for start, end in zip(seq_starts, seq_ends):
            Z_seq = Z_all[start:end]
            T_m = Z_seq.shape[0]

            for local_t in range(T_m):
                if local_t == 0:
                    # No history, cannot form prediction under VAR
                    continue

                # Number of available lags at this position
                L = min(p, local_t)

                # Build prediction: z_hat_t = intercept + sum_{i=1..L} A_i z_{t-i}
                z_hat = intercept.copy()
                for i in range(1, L + 1):
                    z_hat += A_mats[i - 1] @ Z_seq[local_t - i]

                resid = Z_seq[local_t] - z_hat
                sse_resid[start + local_t] = np.sum(resid**2)

        # Attach SSE residuals to DataFrame
        self.df["sse_resid"] = sse_resid

        print(
            "--- Global VAR model (OLS) complete; sse_resid computed for all sequences ---"
        )
        return self

    # Assign sequence identifiers
    def assign_sequence_ids(self):
        """
        Assign sequence identifiers to syllables based on temporal proximity.

        Sequences are defined separately for each 'source_file'. Within each
        file, syllables are sorted by 'onset' time, and a new sequence is
        started whenever the silent gap between the previous syllable's 'offset'
        and the current syllable's 'onset' exceeds the threshold 'seq_bound' in
        seconds.

        Requirements
        ------------
        The DataFrame must contain:
        - 'source_file' : identifier for the audio file.
        - 'onset'       : onset time (in seconds) for each syllable.
        - 'offset'      : offset time (in seconds) for each syllable.
        The configuration must contain:
        - 'seq_bound' : float, maximum allowed silent gap in seconds.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining. A new
            column 'seq_id' is added to the DataFrame.
        """
        if "source_file" not in self.df.columns:
            raise ValueError(
                "assign_sequence_ids requires a 'source_file' column in self.df"
            )
        if "onset" not in self.df.columns or "offset" not in self.df.columns:
            raise ValueError(
                "assign_sequence_ids requires 'onset' and 'offset' columns in self.df"
            )

        if "seq_bound" not in self.config:
            raise ValueError("config must contain a 'seq_bound' key (in seconds)")

        seq_bound = float(self.config["seq_bound"])
        if seq_bound <= 0.0:
            raise ValueError(
                "config['seq_bound'] must be a positive number (in seconds)"
            )

        print(f"--- Assigning seq_id using seq_bound = {seq_bound} seconds ---")

        # Ensure a stable ordering: within each file, sort by onset
        self.df = self.df.sort_values(by=["source_file", "onset"]).reset_index(
            drop=True
        )

        # Store the global sequence identifier across all files
        seq_ids = np.empty(len(self.df), dtype=int)

        current_global_seq_id = 0

        # Group by source_file and process each file separately
        for _, group_idx in self.df.groupby("source_file").groups.items():
            # Extract onset and offset for this file as arrays
            onsets = self.df.loc[group_idx, "onset"].values
            offsets = self.df.loc[group_idx, "offset"].values

            # Initialize per-file sequence identifiers
            file_seq_ids = np.empty(len(group_idx), dtype=int)

            # Start the first sequence for this file
            current_global_seq_id += 1
            file_seq_ids[0] = current_global_seq_id

            # Iterate over subsequent syllables
            for j in range(1, len(group_idx)):
                # Compute silent gap between previous offset and current onset
                gap = onsets[j] - offsets[j - 1]

                if gap > seq_bound:
                    # Start a new global sequence
                    current_global_seq_id += 1

                file_seq_ids[j] = current_global_seq_id

            # Write back into global seq_ids array
            seq_ids[group_idx.to_numpy()] = file_seq_ids

        # Attach seq_id to DataFrame
        self.df["seq_id"] = seq_ids

        print(
            f"--- seq_id assignment complete; total sequences: {self.df['seq_id'].nunique()} ---"
        )
        return self

    # Compute DTW distance
    def compute_dtw_distance(self, seq_id_1, seq_id_2):
        """
        Compute Dynamic Time Warping (DTW) cosine distance between two sequences.

        This method calculates the DTW distance between the latent feature
        sequences of two specified sequence IDs. It first computes a local
        distance matrix using the cosine distance between all pairs of units
        from the two sequences, and then uses this matrix to find the optimal
        alignment path cost with DTW.

        Parameters
        ----------
        seq_id_1 : int
            The identifier for the first sequence.
        seq_id_2 : int
            The identifier for the second sequence.

        Returns
        -------
        float
            The total DTW distance (cost) between the two sequences.

        Raises
        ------
        ValueError
            If 'seq_id' or feature columns are not found in the DataFrame, or
            if one or both of the specified seq_ids do not exist.
        """
        # Validate input
        if "seq_id" not in self.df.columns:
            raise ValueError(
                "DataFrame must contain a 'seq_id' column. Run assign_sequence_ids() first."
            )

        feature_cols = [col for col in self.df.columns if "_feat_" in col]
        if not feature_cols:
            raise ValueError(
                "No feature columns (containing '_feat_') found in the DataFrame."
            )

        # Extract the features for each sequence
        seq1_df = self.df[self.df["seq_id"] == seq_id_1]
        seq2_df = self.df[self.df["seq_id"] == seq_id_2]

        if seq1_df.empty:
            raise ValueError(f"Sequence ID {seq_id_1} not found in the DataFrame.")
        if seq2_df.empty:
            raise ValueError(f"Sequence ID {seq_id_2} not found in the DataFrame.")

        # Extract latent features into NumPy arrays, ensuring temporal order
        features1 = seq1_df.sort_values(by="onset")[feature_cols].values
        features2 = seq2_df.sort_values(by="onset")[feature_cols].values

        # Define cosine distance between two feature vectors
        def _cosine_distance(x, y):
            num = float(np.dot(x, y))
            denom = float(np.linalg.norm(x) * np.linalg.norm(y))
            if denom == 0.0:
                # If one vector is zero, treat distance as maximal (1.0)
                return 1.0
            return 1.0 - num / denom

        # Delegate DTW computation to the external 'dtw' package.
        # The installed 'dtw' package returns a tuple:
        # (distance, costMatrix, accCostMatrix, path)
        distance, _cost, _acc_cost, _path = dtw(
            features1, features2, dist=_cosine_distance
        )

        # Return the DTW distance
        return float(distance)

    # Compute frequency statistics
    def compute_frequency_statistics(self, h5_path, return_traces=False):
        """
        Compute minimum, mean, and maximum frequency statistics for each unit.

        This method loads spectrograms from the HDF5 file and calculates frequency
        statistics for each time bin of each unit. It always updates the internal
        DataFrame in place with summary statistics (global min, mean, max per unit),
        and can optionally return detailed per-time-bin traces.

        Parameters
        ----------
        h5_path : str or Path
            Path to the HDF5 file containing the spectrograms dataset.
        return_traces : bool, optional
            If True, returns a dictionary with detailed per-time-bin frequency
            traces for each unit. If False (default), returns the FeatureProcessor
            instance to support method chaining.

        Returns
        -------
        FeatureProcessor or dict
            If return_traces is False (default), returns self after adding the
            new summary columns ('min_freq', 'mean_freq', 'max_freq',
            'time_bin_ms') to self.df.
            If return_traces is True, returns a dictionary containing:

            - 'time_bins_info': metadata about the time axis
            - 'units': mapping from unit index (h5_index) to per-time-bin traces
              for 'min_freq_trace', 'mean_freq_trace', and 'max_freq_trace'.

        Notes
        -----
        If the metadata contains a 'max_unit_length_s' column (added during
        segmentation), it is used to determine the effective duration represented
        by each spectrogram column. Otherwise, the method falls back to the
        configuration's max unit length settings.
        """
        print("--- Computing frequency statistics from spectrograms ---")

        # Ensure required configuration keys exist
        required_keys = [
            "sr",
            "n_fft",
            "hop_length",
            "n_mels",
            "fmin",
            "fmax",
            "target_shape",
        ]
        for k in required_keys:
            if k not in self.config:
                raise ValueError(f"Config missing required key: '{k}'")

        # Calculate frequency bin centers for the mel spectrogram.
        # We need to reconstruct the mel frequency scale used during generation.
        n_mels = self.config["n_mels"]
        fmin = self.config["fmin"]
        fmax = self.config["fmax"]

        # Compute librosa's mel frequencies
        import librosa

        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

        # Calculate time bin duration in milliseconds for the target shape.
        # The original hop length determines the temporal resolution of the raw spectrogram,
        # But the target shape determines the resolution of the stored spectrogram.

        target_freq_bins = self.config["target_shape"][0]
        target_time_bins = self.config["target_shape"][1]

        # Calculate time per bin in milliseconds.
        # The spectrograms are padded to max_unit_length before being resized to target_time_bins.
        # Therefore, the total duration represented by the x-axis is always max_unit_length.
        # We default to 'simple_max_unit_length' as the reference duration.
        # If 'pykanto_max_unit_length' was used and is different, this might be slightly off,
        # But standard practice is to keep them consistent.
        max_len_s = None

        if (
            "max_unit_length_s" in self.df.columns
            and self.df["max_unit_length_s"].notna().any()
        ):
            max_len_s = float(self.df["max_unit_length_s"].dropna().iloc[0])

        if max_len_s is None:
            # Fall back to configuration, preferring explicit simple/pykanto settings
            if self.config.get("segmentation_method", "").lower() == "pykanto":
                max_len_s = self.config.get("pykanto_max_unit_length")
            elif self.config.get("segmentation_method", "").lower() == "simple":
                max_len_s = self.config.get("simple_max_unit_length")

        if max_len_s is None:
            max_len_s = self.config.get(
                "simple_max_unit_length",
                self.config.get("pykanto_max_unit_length", 0.4),
            )

        # Time resolution is constant for all units
        time_bin_res_ms = (max_len_s * 1000.0) / target_time_bins

        # Since we resized the frequency axis, we need to interpolate the mel frequencies to the new size.
        if target_freq_bins != n_mels:
            # Perform linear interpolation of frequencies to match the new number of bins
            original_indices = np.linspace(0, n_mels - 1, n_mels)
            new_indices = np.linspace(0, n_mels - 1, target_freq_bins)
            freq_scale = np.interp(new_indices, original_indices, mel_freqs)
        else:
            freq_scale = mel_freqs

        # Initialize results structure
        results = {
            "time_bins_info": {
                "count": target_time_bins,
                "resolution_ms": time_bin_res_ms,
                "note": f"Spectrograms are padded to {max_len_s}s and resized. Resolution is constant.",
            },
            "units": {},
        }

        # Prepare lists for DataFrame columns
        global_min_freqs = []
        global_mean_freqs = []
        global_max_freqs = []
        # Fill constant time resolution for column
        time_resolution_ms = [time_bin_res_ms] * len(self.df)

        # Iterate through all units in the DataFrame
        if "h5_index" not in self.df.columns:
            raise ValueError("DataFrame must contain 'h5_index' column.")

        with h5py.File(h5_path, "r", libver="latest", swmr=True) as hf:
            dataset = hf["spectrograms"]

            # Iterate with progress bar
            for idx, row in tqdm(
                self.df.iterrows(), total=len(self.df), desc="Analyzing spectrograms"
            ):
                h5_idx = int(row["h5_index"])

                # Load spectrogram
                spec_data = dataset[h5_idx]
                if spec_data.dtype == np.uint8:
                    spec = spec_data.astype(np.float32) / 255.0
                else:
                    spec = spec_data[...]

                # Define a threshold for "significant energy" relative to the peak in that time bin
                threshold = FREQ_ENERGY_THRESHOLD

                min_trace = []
                max_trace = []
                mean_trace = []

                for t in range(spec.shape[1]):
                    col = spec[:, t]

                    # If the column is effectively silent
                    if np.max(col) < 0.01:
                        min_trace.append(np.nan)
                        max_trace.append(np.nan)
                        mean_trace.append(np.nan)
                        continue

                    # Find indices with energy above threshold
                    col_thresh = np.max(col) * threshold
                    active_indices = np.where(col >= col_thresh)[0]

                    if len(active_indices) == 0:
                        min_trace.append(np.nan)
                        max_trace.append(np.nan)
                        mean_trace.append(np.nan)
                        continue

                    # Determine minimum and maximum frequency indices
                    min_idx = active_indices[0]
                    max_idx = active_indices[-1]

                    min_f = freq_scale[min_idx]
                    max_f = freq_scale[max_idx]

                    # Compute weighted mean (centroid)
                    active_energies = col[active_indices]
                    active_freqs = freq_scale[active_indices]

                    if np.sum(active_energies) > 0:
                        mean_f = np.average(active_freqs, weights=active_energies)
                    else:
                        mean_f = np.nan

                    min_trace.append(float(min_f))
                    max_trace.append(float(max_f))
                    mean_trace.append(float(mean_f))

                # Store detailed traces
                results["units"][h5_idx] = {
                    "min_freq_trace": min_trace,
                    "mean_freq_trace": mean_trace,
                    "max_freq_trace": max_trace,
                }

                # Compute global statistics for this unit (ignoring NaNs)
                valid_mins = [x for x in min_trace if not np.isnan(x)]
                valid_means = [x for x in mean_trace if not np.isnan(x)]
                valid_maxs = [x for x in max_trace if not np.isnan(x)]

                global_min_freqs.append(np.min(valid_mins) if valid_mins else np.nan)
                global_mean_freqs.append(
                    np.mean(valid_means) if valid_means else np.nan
                )
                global_max_freqs.append(np.max(valid_maxs) if valid_maxs else np.nan)

        # Add columns to DataFrame
        self.df["min_freq"] = global_min_freqs
        self.df["mean_freq"] = global_mean_freqs
        self.df["max_freq"] = global_max_freqs
        self.df["time_bin_ms"] = time_resolution_ms

        print("--- Frequency statistics computation complete ---")

        if return_traces:
            return results

        return self

    # Plot BIRCH SSE elbow
    def plot_birch_sse_elbow(self, k_range):
        """
        Plot the sum of squared errors for a range of cluster counts in BIRCH clustering.

        This method computes and visualizes the sum of squared errors (SSE)
        corresponding to different numbers of clusters in BIRCH clustering.
        It helps identify an appropriate number of clusters using the elbow
        method.

        Parameters
        ----------
        k_range : iterable of int
            Iterable (for example, a list or range) of cluster counts 'k' to
            evaluate.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining.
        """
        # Apply the plot style from the configuration
        set_plot_style(SimpleNamespace(**self.config))

        # Automatically infer feature columns
        feature_cols = [col for col in self.df.columns if col.startswith("pacmap_")]

        if not feature_cols:
            raise ValueError(
                "No PaCMAP columns starting with 'pacmap_' found in the DataFrame"
            )

        # Extract features to be used for clustering
        features = self.df[feature_cols].values
        sse_values = []

        print("--- Calculating SSE for BIRCH clustering to find elbow point ---")

        for k in tqdm(k_range, desc="Testing k values"):
            # Initialize and fit the BIRCH model
            birch_model = Birch(n_clusters=k)
            birch_model.fit(features)

            # Manually calculate the sum of squared errors (SSE)
            final_centroids = birch_model.subcluster_centers_
            labels = birch_model.labels_
            current_sse = 0.0

            # Sum the squared distances of samples to their closest cluster center
            for i in range(k):
                points_in_cluster = features[labels == i]
                if points_in_cluster.shape[0] > 0:
                    centroid_for_cluster = final_centroids[i]
                    current_sse += np.sum(
                        (points_in_cluster - centroid_for_cluster) ** 2
                    )
            sse_values.append(current_sse)

        # Create the elbow plot
        plt.figure(figsize=(12, 4))
        line_color = "white" if self.config.get("dark_mode", False) else "black"
        plt.plot(list(k_range), sse_values, "o-", color=line_color)
        plt.xlabel("Number of clusters (k)", fontsize=12)
        plt.ylabel("Sum of squared errors (SSE)", fontsize=12)
        plt.title("Elbow method for optimal k in BIRCH clustering", fontsize=14)
        plt.xticks(list(k_range), rotation=45)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

        return self

    # Interactive embedding plot (exported as HTML)
    def interactive_embedding_plot(
        self,
        h5_path,
        output_html_path,
        thumb_size=96,
        point_alpha=0.7,
        point_size=3,
    ):
        """
        Export a self-contained HTML file with an interactive embedding plot.

        This method creates a standalone HTML file that uses Plotly.js in the
        browser (no Python backend required) to display the PaCMAP embedding.
        Hovering over points in the scatterplot updates a grid of pre-rendered
        spectrogram thumbnails for the focal unit and its nearest neighbors.

        All required data (coordinates, neighbor indices, and spectrogram
        thumbnails encoded as base64 PNGs) are embedded directly into the HTML
        file so it can be shared and opened without any additional files.

        Parameters
        ----------
        h5_path : str or Path
            Path to the HDF5 file containing the spectrograms dataset.
        output_html : str or Path
            Path at which to save the resulting HTML file.
        thumb_size : int, optional
            Approximate size (in pixels) of the square spectrogram thumbnails.
            The default is 96.
        point_alpha : float, optional
            Opacity of the scatter points (0.0 to 1.0). The default is 0.7.
        point_size : int, optional
            Size of the scatter points in pixels. The default is 3.

        Returns
        -------
        FeatureProcessor
            The current instance, returned to enable method chaining.
        """
        h5_path = Path(h5_path)
        output_html_path = Path(output_html_path)

        if "pacmap_x" not in self.df.columns or "pacmap_y" not in self.df.columns:
            raise ValueError(
                "interactive_embedding_plot requires 'pacmap_x' and "
                "'pacmap_y' columns. Run run_pacmap() first."
            )

        if "h5_index" not in self.df.columns:
            raise ValueError(
                "interactive_embedding_plot requires an 'h5_index' "
                "column to locate spectrograms in the HDF5 file."
            )

        coords = self.df[["pacmap_x", "pacmap_y"]].to_numpy()
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                "PaCMAP coordinates must be an (n, 2) array from "
                "columns 'pacmap_x' and 'pacmap_y'"
            )

        n_points = coords.shape[0]
        if n_points == 0:
            raise ValueError("No points found in DataFrame for embedding.")

        # Build neighbor graph (indices of nearest neighbors per point)
        # Always use 6: one focal (F) + 5 neighbors (N1..N5) for display.
        k = min(6, n_points)
        tree = cKDTree(coords)
        _, neighbor_indices = tree.query(coords, k=k)
        if k == 1:
            neighbor_indices = neighbor_indices.reshape(-1, 1)

        # Compute density background grid similar to the static plot
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        plot_xmin = x_min - x_range * 0.05
        plot_xmax = x_max + x_range * 0.05
        plot_ymin = y_min - y_range * 0.05
        plot_ymax = y_max + y_range * 0.05

        # Note: histogram2d expects (x, y); we pass (x, y) explicitly so that
        # x_edges correspond to the first (x) dimension and y_edges to the
        # second (y) dimension used in the heatmap.
        counts, x_edges, y_edges = np.histogram2d(
            coords[:, 0],
            coords[:, 1],
            bins=(DENSITY_HISTOGRAM_BINS, DENSITY_HISTOGRAM_BINS),
            range=[[plot_xmin, plot_xmax], [plot_ymin, plot_ymax]],
        )
        z = gaussian_filter(counts, sigma=8)
        z = np.log1p(z)

        # Transpose z so that the first dimension corresponds to y and the second to x,
        # which is how Plotly heatmaps interpret the z matrix.
        z = z.T

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        # Pre-render spectrogram thumbnails and encode as base64 PNGs
        print("--- Encoding spectrogram thumbnails for HTML export ---")
        thumbs: list[str] = []
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as hf:
            dataset = hf["spectrograms"]
            for _, row in tqdm(
                self.df.iterrows(),
                total=len(self.df),
                desc="Thumbnails",
            ):
                h5_idx = int(row["h5_index"])
                spec_data = dataset[h5_idx]
                if spec_data.dtype == np.uint8:
                    spec = spec_data.astype(np.float32) / 255.0
                else:
                    spec = spec_data[...]

                # Render spectrogram to a small PNG in memory
                buf = io.BytesIO()
                plt.imsave(
                    buf,
                    spec,
                    cmap="viridis",
                    origin="lower",
                    format="png",
                    dpi=thumb_size,
                )
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("ascii")
                thumbs.append(f"data:image/png;base64,{b64}")

        # Optional: embed project logo (if available) as base64
        logo_b64 = None
        try:
            root = Path(__file__).resolve().parents[2]
            logo_path = root / "docs" / "_static" / "logo.png"
            if logo_path.exists():
                with logo_path.open("rb") as f:
                    logo_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception:
            logo_b64 = None

        data = {
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "neighbors": neighbor_indices.tolist(),
            "thumbs": thumbs,
            "logo": logo_b64,
            "density": {
                "z": z.tolist(),
                "x": x_centers.tolist(),
                "y": y_centers.tolist(),
            },
        }

        data_json = json.dumps(data)

        # Minimal HTML + JS using Plotly via CDN
        html = f"""<!DOCTYPE html>
                <html lang="en">
                <head>
                <meta charset="utf-8" />
                <title>Chatter Interactive Embedding</title>
                <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
                <style>
                    body {{
                    margin: 0;
                    background-color: #440154;
                    color: #fff;
                    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    }}
                    #container {{
                    display: flex;
                    flex-direction: row;
                    height: 100vh;
                    width: 100vw;
                    box-sizing: border-box;
                    padding: 8px;
                    gap: 8px;
                    }}
                    #scatter-wrapper {{
                    position: relative;
                    flex: 3;
                    overflow: hidden;
                    }}
                    #scatter {{
                    width: 100%;
                    height: 100%;
                    }}
                    #thumbs {{
                    flex: 2;
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    grid-auto-rows: minmax({thumb_size}px, auto);
                    gap: 8px;
                    align-content: flex-start;
                    }}
                    .thumb-wrapper {{
                    position: relative;
                    border: 1px solid #fff;
                    }}
                    .thumb-wrapper img {{
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                    display: block;
                    }}
                    .thumb-label {{
                    position: absolute;
                    top: 4px;
                    left: 4px;
                    padding: 3px 6px;
                    background-color: rgba(0, 0, 0, 0.6);
                    color: #fff;
                    font-size: 13px;
                    border-radius: 4px;
                    }}
                    #logo-overlay {{
                    position: absolute;
                    top: 8px;
                    left: 8px;
                    width: 200px;
                    height: auto;
                    z-index: 10;
                    opacity: 1;
                    }}
                </style>
                </head>
                <body>
                <div id="container">
                    <div id="scatter-wrapper">
                    <img id="logo-overlay" style="display:none;" />
                    <div id="scatter"></div>
                    </div>
                    <div id="thumbs"></div>
                </div>
                <script>
                    const DATA = {data_json};

                    const scatterDiv = document.getElementById('scatter');
                    const thumbsDiv = document.getElementById('thumbs');
                    const logoImg = document.getElementById('logo-overlay');

                    if (DATA.logo) {{
                    logoImg.src = 'data:image/png;base64,' + DATA.logo;
                    logoImg.style.display = 'block';
                    }}

                    // Density background heatmap
                    const heatmap = {{
                    x: DATA.density.x,
                    y: DATA.density.y,
                    z: DATA.density.z,
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    showscale: false,
                    zsmooth: 'best',
                    hoverinfo: 'skip',
                    }};

                    // Scatter overlay
                    const trace = {{
                    x: DATA.x,
                    y: DATA.y,
                    mode: 'markers',
                    type: 'scattergl',
                    marker: {{
                        size: {point_size},
                        color: 'rgba(255,255,255,{point_alpha})',
                        opacity: {point_alpha},
                    }},
                    hoverinfo: 'none',
                    }};

                    const layout = {{
                    // Viridis start color (#440154FF) as background
                    paper_bgcolor: '#440154',
                    plot_bgcolor: '#440154',
                    xaxis: {{
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false,
                        ticks: "",
                        fixedrange: true,
                    }},
                    yaxis: {{
                        showgrid: false,
                        zeroline: false,
                        showticklabels: false,
                        ticks: "",
                        fixedrange: true,
                    }},
                    margin: {{l: 10, r: 10, t: 10, b: 10}},
                    dragmode: false,
                    }};

                    Plotly.newPlot(scatterDiv, [heatmap, trace], layout, {{
                    responsive: true,
                    displayModeBar: false,
                    }});

                    function showThumbs(centerIdx) {{
                    const neighbors = DATA.neighbors[centerIdx];
                    thumbsDiv.innerHTML = '';

                    const labels = ['Focal'].concat(
                        neighbors.slice(1).map((_, i) => 'Neighbor ' + (i + 1))
                    );

                    neighbors.forEach((idx, j) => {{
                        const wrapper = document.createElement('div');
                        wrapper.className = 'thumb-wrapper';

                        const img = document.createElement('img');
                        img.src = DATA.thumbs[idx];
                        wrapper.appendChild(img);

                        const label = document.createElement('div');
                        label.className = 'thumb-label';
                        label.textContent = labels[j];
                        wrapper.appendChild(label);

                        thumbsDiv.appendChild(wrapper);
                    }});
                    }}

                    scatterDiv.on('plotly_hover', function(ev) {{
                    if (!ev.points || !ev.points.length) return;
                    const idx = ev.points[0].pointIndex;
                    showThumbs(idx);
                    }});
                </script>
                </body>
                </html>
                """

        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        output_html_path.write_text(html, encoding="utf-8")
        print(f"--- Interactive HTML embedding saved to {output_html_path} ---")

        return self

    # Static embedding plot
    def static_embedding_plot(
        self,
        h5_path,
        output_path=None,
        seed=42,
        focal_quantile=0.8,
        point_alpha=0.3,
        point_size=2,
        margin=0.02,
        zoom_padding=0.05,
        num_neighbors=3,
    ):
        """
        Create a publication-quality static plot of the embedding space.

        This method generates a visualization that includes a 2D density map of
        the embedding and four "callouts" showing focal syllables and their
        nearest neighbors. Focal points are selected automatically from the
        fringes of each quadrant of the embedding space to ensure a representative
        sample of unique points. The plot is designed to be border-free with a
        seamless viridis background.

        Parameters
        ----------
        h5_path : str or Path
            Path to the HDF5 file containing the spectrograms dataset.
        output_path : str or Path, optional
            Path to save the final PNG image. If None, the plot is displayed
            directly using plt.show(). The default is None.
        seed : int, optional
            Seed for the random number generator to ensure reproducible selection
            of focal points. The default is 42.
        point_alpha : float, optional
            The alpha (transparency) of the scatter points in the background.
            The default is 0.3.
        point_size : int or float, optional
            The size of the scatter points in the background. The default is 2.
        margin : float, optional
            The margin from the plot edge to the nearest edge of a callout
            group, as a fraction of the plot's total width/height. The default
            is 0.02.
        zoom_padding : float, optional
            The padding to add around the data points as a percentage of the
            data's range, effectively controlling the zoom level. The default
            is 0.05 (5%).
        num_neighbors : int, optional
            The number of nearest neighbors to display for each focal point.
            The default is 3.

        """
        # Ensure required columns exist
        required_cols = ["pacmap_x", "pacmap_y", "h5_index"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Dataframe is missing required column: '{col}'")

        # Automatically select focal points from quadrants
        print(
            f"--- Automatically selecting focal points from quadrants with seed {seed} ---"
        )
        rng = np.random.default_rng(seed)
        x_mean = self.df["pacmap_x"].mean()
        y_mean = self.df["pacmap_y"].mean()

        # Define quadrants
        quadrants = {
            "top_left": self.df[
                (self.df["pacmap_x"] <= x_mean) & (self.df["pacmap_y"] > y_mean)
            ],
            "top_right": self.df[
                (self.df["pacmap_x"] > x_mean) & (self.df["pacmap_y"] > y_mean)
            ],
            "bottom_left": self.df[
                (self.df["pacmap_x"] <= x_mean) & (self.df["pacmap_y"] <= y_mean)
            ],
            "bottom_right": self.df[
                (self.df["pacmap_x"] > x_mean) & (self.df["pacmap_y"] <= y_mean)
            ],
        }

        # Sample one point from the fringes of each quadrant
        focal_indices = []
        quadrant_order = ["top_left", "top_right", "bottom_left", "bottom_right"]

        for q_name in quadrant_order:
            quadrant_df = quadrants[q_name]
            if quadrant_df.empty:
                raise ValueError(
                    f"Cannot select a focal point: the {q_name.replace('_', ' ')} quadrant is empty"
                )

            # Calculate distance from the center for each point in the quadrant
            distances = np.sqrt(
                (quadrant_df["pacmap_x"] - x_mean) ** 2
                + (quadrant_df["pacmap_y"] - y_mean) ** 2
            )

            # Find the 95th percentile distance to identify the fringe
            fringe_threshold = distances.quantile(focal_quantile)
            fringe_points = quadrant_df[distances >= fringe_threshold]

            # If the fringe is empty (for example, very few points), fall back to the whole quadrant
            if fringe_points.empty:
                fringe_points = quadrant_df

            # Sample from the fringe points
            focal_indices.append(fringe_points.sample(1, random_state=rng).index[0])

        # Get nearest neighbors for all points
        print("--- Finding nearest neighbors ---")
        coords = self.df[["pacmap_x", "pacmap_y"]].values
        nn = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm="auto").fit(
            coords
        )
        _, indices = nn.kneighbors(coords)
        neighbor_indices = indices[:, 1:]

        # Set up the plot
        print("--- Creating the plot ---")
        fig, ax = plt.subplots(figsize=(16, 9))

        # Get the starting color of viridis and set it as the background
        viridis_start_color = plt.colormaps["viridis"](0)
        fig.patch.set_facecolor(viridis_start_color)
        ax.set_facecolor(viridis_start_color)

        # Calculate plot boundaries with zoom padding
        x_min, x_max = self.df["pacmap_x"].min(), self.df["pacmap_x"].max()
        y_min, y_max = self.df["pacmap_y"].min(), self.df["pacmap_y"].max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        plot_xmin = x_min - x_range * zoom_padding
        plot_xmax = x_max + x_range * zoom_padding
        plot_ymin = y_min - y_range * zoom_padding
        plot_ymax = y_max + y_range * zoom_padding

        # Create two-dimensional density background
        print("--- Plotting density background (using fast 2d histogram) ---")
        counts, _, _ = np.histogram2d(
            self.df["pacmap_y"],
            self.df["pacmap_x"],
            bins=(DENSITY_HISTOGRAM_BINS, DENSITY_HISTOGRAM_BINS),
            range=[[plot_ymin, plot_ymax], [plot_xmin, plot_xmax]],
        )
        z = gaussian_filter(counts, sigma=8)
        z = np.log1p(z)

        ax.imshow(
            z,
            cmap="viridis",
            extent=[plot_xmin, plot_xmax, plot_ymin, plot_ymax],
            aspect="auto",
            origin="lower",
        )

        # Add all points and set limits
        ax.scatter(
            self.df["pacmap_x"],
            self.df["pacmap_y"],
            s=point_size,
            color="white",
            alpha=point_alpha,
            marker="o",
            edgecolors="none",
        )
        ax.set_xlim(plot_xmin, plot_xmax)
        ax.set_ylim(plot_ymin, plot_ymax)
        ax.axis("off")

        # Open HDF5 file to load spectrograms
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as hf:

            def _load_spec(h5_idx):
                spec_data = hf["spectrograms"][h5_idx]
                return (
                    (spec_data.astype(np.float32) / 255.0)
                    if spec_data.dtype == np.uint8
                    else spec_data[...]
                )

            # Calculate positions and add spectrogram callouts
            print("--- Calculating callout positions and adding spectrograms ---")
            plot_width = plot_xmax - plot_xmin
            plot_height = plot_ymax - plot_ymin

            # Calculate correction factor to ensure images maintain aspect ratio
            # Given the unequal data scales and figure dimensions
            fig_w, fig_h = fig.get_size_inches()
            correction_factor = (plot_height / plot_width) * (fig_w / fig_h)

            # Get spectrogram aspect ratio from a sample
            sample_spec = _load_spec(int(self.df.loc[focal_indices[0], "h5_index"]))
            spec_height, spec_width = sample_spec.shape
            aspect_ratio = spec_width / spec_height if spec_height > 0 else 1.0

            # Define image sizes by constraining the total width of the neighbor group
            max_group_width = plot_width * 0.4  # Constrain group to 40% of plot width

            neighbor_img_width = max_group_width / num_neighbors
            neighbor_img_height = (
                neighbor_img_width * correction_factor
            ) / aspect_ratio

            # Make focal images slightly larger than neighbors
            focal_img_width = neighbor_img_width * 1.3
            focal_img_height = (focal_img_width * correction_factor) / aspect_ratio

            # Calculate symmetrical positions based on the full plot dimensions
            margin_x = plot_width * margin
            margin_y = plot_height * margin

            # Calculate base x positions for the focal spectrograms
            left_focal_x = plot_xmin + margin_x + (max_group_width / 2)
            right_focal_x = plot_xmax - margin_x - (max_group_width / 2)

            # Calculate base y positions for the neighbor groups
            top_neighbor_y = plot_ymax - margin_y - (neighbor_img_height / 2)
            bottom_neighbor_y = plot_ymin + margin_y + (neighbor_img_height / 2)

            # Vertical offset between focal and neighbor centers
            vertical_padding = plot_height * 0.02
            neighbor_y_offset = (
                focal_img_height / 2 + neighbor_img_height / 2 + vertical_padding
            )

            callout_positions = [
                {"x": left_focal_x, "y": top_neighbor_y - neighbor_y_offset},
                {"x": right_focal_x, "y": top_neighbor_y - neighbor_y_offset},
                {"x": left_focal_x, "y": bottom_neighbor_y + neighbor_y_offset},
                {"x": right_focal_x, "y": bottom_neighbor_y + neighbor_y_offset},
            ]
            neighbor_vertical_positions = [True, True, False, False]

            def draw_image_with_border(
                ax,
                spec_data,
                x,
                y,
                w,
                h,
                border_color="white",
                border_width=1,
                zorder=10,
            ):
                ax.imshow(
                    spec_data,
                    extent=[x - w / 2, x + w / 2, y - h / 2, y + h / 2],
                    aspect="auto",
                    cmap="viridis",
                    origin="lower",
                    interpolation="antialiased",
                    zorder=zorder,
                )
                rect = patches.Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    linewidth=border_width,
                    edgecolor=border_color,
                    facecolor="none",
                    transform=ax.transData,
                    clip_on=False,
                    zorder=zorder + 1,
                )
                ax.add_patch(rect)

            # Add callout groups
            for i, focal_idx in enumerate(focal_indices):
                focal_x, focal_y = self.df.loc[focal_idx, ["pacmap_x", "pacmap_y"]]
                callout_pos = callout_positions[i]

                # Add dashed line with a low zorder to be in the back
                ax.plot(
                    [focal_x, callout_pos["x"]],
                    [focal_y, callout_pos["y"]],
                    color="white",
                    linestyle="--",
                    linewidth=1,
                    alpha=1,
                    zorder=5,
                )

                # Draw focal spectrogram with a high zorder
                focal_h5_idx = int(self.df.loc[focal_idx, "h5_index"])
                focal_spec = _load_spec(focal_h5_idx)
                draw_image_with_border(
                    ax,
                    focal_spec,
                    callout_pos["x"],
                    callout_pos["y"],
                    focal_img_width,
                    focal_img_height,
                    border_width=2,
                    zorder=10,
                )

                # Determine vertical placement of neighbor spectrograms
                neighbors_above = neighbor_vertical_positions[i]
                neighbor_y_center = callout_pos["y"] + (
                    neighbor_y_offset * (1 if neighbors_above else -1)
                )

                # Calculate starting x-position to center the neighbor group
                neighbor_start_x = callout_pos["x"] - (
                    neighbor_img_width * (num_neighbors - 1) / 2.0
                )

                # Draw neighbor spectrograms with a high zorder
                current_focal_neighbor_indices = neighbor_indices[
                    self.df.index.get_loc(focal_idx)
                ]

                for j, neighbor_idx in enumerate(current_focal_neighbor_indices):
                    neighbor_x = neighbor_start_x + j * neighbor_img_width
                    neighbor_h5_idx = int(self.df.loc[neighbor_idx, "h5_index"])
                    neighbor_spec = _load_spec(neighbor_h5_idx)
                    draw_image_with_border(
                        ax,
                        neighbor_spec,
                        neighbor_x,
                        neighbor_y_center,
                        neighbor_img_width,
                        neighbor_img_height,
                        border_width=1,
                        zorder=10,
                    )

        # Save or show the plot
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"--- Saving plot to {output_path} ---")
            fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
        else:
            print("--- Displaying plot ---")
            plt.show()

        plt.close(fig)

        return self
