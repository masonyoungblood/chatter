# Silence annoying warning
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated as an API.*"
)

# Import necessary libraries
import os  # noqa: E402
import random  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from pathlib import Path  # noqa: E402
from tqdm import tqdm  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from transformers import AutoImageProcessor, AutoModel  # noqa: E402

# Import local modules
from .models import Encoder, ae_loss  # noqa: E402
from .data import SpectrogramDataset  # noqa: E402
from .config import set_plot_style  # noqa: E402


# Trainer class
class Trainer:
    """
    Trainer class for the unified variational autoencoder implementation.

    This class manages model initialization, training, evaluation, and feature
    extraction for the shared :class:`~chatter.models.Encoder`, which can be
    configured as either a convolutional or vector-based VAE. It is designed to
    operate on a single device (CPU, CUDA GPU, or Apple MPS).

    Attributes
    ----------
    config : dict
        Configuration dictionary containing model and training parameters.
    device : torch.device
        Computation device used for training and inference.
    ae_type : str
        Type of autoencoder architecture ('convolutional' or 'vector').
    ae_model : Encoder
        Unified variational autoencoder model encapsulating encoder and decoder components.
    """

    # Initialize trainer
    def __init__(self, config):
        """
        Initialize the Trainer with a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing model and training parameters.
        """
        # Store configuration
        self.config = dict(config)

        # Determine device for computation
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu",
        )

        # Reduce CPU thread contention for better parallel DataLoader performance
        torch.set_num_threads(1)

        # Enable cuDNN autotuner for faster training on CUDA
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Extract model configuration parameters
        self.ae_type = self.config.get("ae_type", "convolutional")

        print(f"Initializing {self.ae_type} variational autoencoder")

        # Initialize model based on architecture type
        if self.ae_type == "vector":
            h, w = self.config["target_shape"]
            input_dim = h * w
            self.ae_model = Encoder(
                ae_type="vector",
                latent_dim=self.config["latent_dim"],
                input_dim=input_dim,
                target_shape=self.config["target_shape"],
            )
        else:
            self.ae_model = Encoder(
                ae_type="convolutional",
                latent_dim=self.config["latent_dim"],
                target_shape=self.config["target_shape"],
            )

        # Move model to device
        self.ae_model.to(self.device)
        print(f"Using device: {self.device}")

    # Create trainer from pre-trained model
    @classmethod
    def from_trained(cls, config, model_dir):
        """
        Create a Trainer instance and load a pre-trained model.

        This class method instantiates a new Trainer with the provided
        configuration and immediately loads model weights from the specified
        path. It enables direct use of methods such as 'extract_and_save_features'
        or 'plot_reconstructions' without retraining.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the model and training.
        model_dir : str or Path
            Path to the saved model directory.

        Returns
        -------
        Trainer
            An instance of the Trainer class with the model weights loaded.
        """
        print(f"Instantiating Trainer from pre-trained model at {model_dir}...")

        trainer_instance = cls(config)
        trainer_instance.load_ae(model_dir)

        return trainer_instance

    # Train autoencoder
    def train_ae(self, unit_df, h5_path, model_dir, subset=None):
        """
        Train the variational autoencoder using an HDF5 dataset.

        This method creates a SpectrogramDataset from an HDF5 file, constructs
        a DataLoader, and runs a standard training loop for a configured number
        of epochs. It optionally trains on a random subset of units, and saves
        the trained model and loss history to disk.

        Parameters
        ----------
        unit_df : pd.DataFrame
            DataFrame containing unit metadata with a column 'h5_index'
            referring to indices in the HDF5 'spectrograms' dataset.
        h5_path : str or Path
            Path to the HDF5 file containing spectrograms.
        model_dir : str or Path
            Directory in which to save the trained model and loss history CSV.
        subset : float, optional
            Proportion of units to use for training. Must be in the range
            (0, 1) if provided. If None or outside this range, the full
            dataset is used. The default is None.

        Returns
        -------
        None
            This method trains a model and writes the results to disk but does
            not return a value.
        """
        # Convert model directory to pathlib object
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        model_save_path = model_dir / "model.pth"
        loss_output_path = model_dir / "loss.csv"

        # Get all available indices from DataFrame
        all_indices = unit_df["h5_index"].tolist()

        if not all_indices:
            print("No data found in the DataFrame")
            return

        # Handle subsetting for training
        if subset is not None and 0.0 < subset < 1.0:
            num_files = len(all_indices)
            subset_size = max(1, int(num_files * subset))
            subset_size = min(subset_size, num_files)
            subset_pct = subset * 100.0
            print(
                f"--- Training on a random subset of {subset_size} units ({subset_pct:.1f}%) ---"
            )
            training_indices = random.sample(all_indices, subset_size)
        else:
            print(f"--- Training on the full dataset of {len(all_indices)} units ---")
            training_indices = all_indices

        # Create dataset and DataLoader
        dataset = SpectrogramDataset(h5_path, training_indices)
        num_workers = min(4, os.cpu_count() or 1)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=self.device.type == "cuda",
        )

        # Initialize optimizer
        optimizer = optim.AdamW(self.ae_model.parameters(), lr=self.config["lr"])
        loss_history = []

        print(
            f"\nStarting training for {self.config['epochs']} epochs using {num_workers} DataLoader workers..."
        )

        # Training loop with progress bar
        epoch_pbar = tqdm(range(self.config["epochs"]), desc="Training model")

        for epoch in epoch_pbar:
            self.ae_model.train()
            total_loss = 0.0

            for batch in dataloader:
                # Move batch to device
                if self.device.type == "cuda":
                    x = batch.to(self.device, non_blocking=True)
                else:
                    x = batch.to(self.device)

                # Perform forward pass and loss computation
                optimizer.zero_grad(set_to_none=True)

                x_recon, mu, log_var = self.ae_model(x)

                loss = ae_loss(
                    x,
                    x_recon,
                    mu,
                    log_var,
                    beta=self.config.get("beta", 1.0),
                )

                # Perform backward pass and optimization
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Record epoch loss
            avg_loss = total_loss / max(1, len(dataloader))
            loss_history.append({"epoch": epoch + 1, "average_loss": avg_loss})
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")

        # Save trained model
        torch.save(self.ae_model.state_dict(), model_save_path)
        print(f"--- Training complete. Model saved to {model_save_path} ---")

        # Save loss history
        loss_df = pd.DataFrame(loss_history)
        loss_df.to_csv(loss_output_path, index=False)
        print(f"Loss history saved to {loss_output_path}")

    # Load pre-trained weights
    def load_ae(self, model_dir):
        """
        Load pre-trained weights into the variational autoencoder model.

        Parameters
        ----------
        model_dir : str or Path
            Path to the saved model directory.

        Returns
        -------
        None
            This method loads model weights and sets the model to evaluation
            mode. It prints status messages and does not return a value.
        """
        model_path = Path(model_dir) / "model.pth"

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.ae_model.load_state_dict(state_dict)
            self.ae_model.eval()
            print(f"Successfully loaded pre-trained model from {model_path}")

        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Model not loaded")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    # Plot reconstructions
    def plot_reconstructions(self, unit_df, h5_path, num_examples=8):
        """
        Plot a side-by-side comparison of original and reconstructed spectrograms.

        This method samples a set of unit spectrograms from the HDF5 dataset,
        passes them through the autoencoder, and visualizes the original and
        reconstructed spectrograms for qualitative inspection of model
        performance.

        Parameters
        ----------
        unit_df : pd.DataFrame
            DataFrame containing unit metadata with a column 'h5_index'
            referring to indices in the HDF5 'spectrograms' dataset.
        h5_path : str or Path
            Path to the HDF5 file containing spectrograms.
        num_examples : int, optional
            Number of examples to plot. If the dataset contains fewer than
            'num_examples' units, all available units are plotted. The default
            is 8.

        Returns
        -------
        None
            This method displays a matplotlib figure and does not return a value.
        """
        # Apply the plot style from the configuration
        set_plot_style(SimpleNamespace(**self.config))

        # Get all available indices
        all_indices = unit_df["h5_index"].tolist()

        if not all_indices:
            print("No data found in the DataFrame for plotting")
            return

        # Sample random spectrograms
        num_to_sample = min(num_examples, len(all_indices))
        sampled_indices = random.sample(all_indices, num_to_sample)
        dataset = SpectrogramDataset(h5_path, sampled_indices)
        data = torch.stack([dataset[i] for i in range(len(dataset))])

        # Get reconstructions from model
        self.ae_model.eval()
        with torch.no_grad():
            x_img = data.to(self.device)
            x_recon_img = self.ae_model(x_img)[0]

        # Prepare images for plotting
        originals = x_img.cpu().squeeze(1)
        reconstructions = x_recon_img.cpu().squeeze(1)

        # Calculate figure size based on spectrogram aspect ratio
        if num_to_sample > 0:
            spec_height, spec_width = originals[0].shape
            aspect_ratio = spec_width / spec_height if spec_height > 0 else 1.0
        else:
            aspect_ratio = 1.0

        row_height = 2.0
        fig_width = row_height * aspect_ratio * num_to_sample
        fig_height = row_height * 2

        # Create comparison plot
        fig, axes = plt.subplots(2, num_to_sample, figsize=(fig_width, fig_height))

        if num_to_sample == 1:
            axes = np.array(axes).reshape(2, 1)

        for i in range(num_to_sample):
            axes[0, i].imshow(
                originals[i], cmap="viridis", origin="lower", aspect="auto"
            )
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            axes[1, i].imshow(
                reconstructions[i],
                cmap="viridis",
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

            if i == 0:
                axes[0, i].set_ylabel("Original", fontsize=12)
                axes[1, i].set_ylabel("Reconstruction", fontsize=12)

        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
        plt.show()

    # Extract and save latent features
    def extract_and_save_features(self, unit_df, h5_path, model_dir, output_csv_path):
        """
        Extract latent features for all units using the HDF5 file and save them.

        This method loads a trained autoencoder model, iterates through all
        spectrograms in the HDF5 dataset, encodes them into latent features,
        and writes a combined DataFrame containing both metadata and latent
        features to CSV.

        Parameters
        ----------
        unit_df : pd.DataFrame
            DataFrame containing unit metadata with a column 'h5_index'
            referring to indices in the HDF5 'spectrograms' dataset.
        h5_path : str or Path
            Path to the HDF5 file containing spectrograms.
        model_dir : str or Path
            Path to the saved model directory.
        output_csv_path : str or Path
            Path to the CSV file in which to store metadata and latent features.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing metadata and extracted features for all units,
            or None if the model could not be loaded successfully.
        """
        print("\n--- Starting feature extraction ---")

        # Load trained model
        try:
            self.load_ae(model_dir)
        except Exception as e:
            print(f"Could not load model for feature extraction: {e}")
            return None

        # Use a stable order (sorted by h5_index) to keep metadata aligned with features
        unit_df_sorted = unit_df.sort_values("h5_index").reset_index(drop=True)
        all_indices = unit_df_sorted["h5_index"].tolist()

        dataset = SpectrogramDataset(h5_path, all_indices)

        # Create DataLoader for batch processing
        num_workers = min(4, os.cpu_count() or 1)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"] * 2,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=self.device.type == "cuda",
        )

        # Extract latent features in batches
        all_features = []
        self.ae_model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if self.device.type == "cuda":
                    x = batch.to(self.device, non_blocking=True)
                else:
                    x = batch.to(self.device)

                features, _ = self.ae_model.encode(x)
                all_features.append(features.cpu())

        # Concatenate features and create DataFrame
        latent_features = torch.cat(all_features, dim=0).numpy()
        feature_cols = [f"ae_feat_{i}" for i in range(self.config["latent_dim"])]
        feature_df = pd.DataFrame(latent_features, columns=feature_cols)

        # Combine metadata with features
        result_df = pd.concat([unit_df_sorted, feature_df], axis=1)

        # Save final CSV
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv_path, index=False)
        print(
            f"\n--- Pipeline complete. Exported data for {len(result_df)} units to {output_csv_path} ---"
        )

        return result_df

    # Extract and save computer vision features
    def extract_and_save_comp_viz_features(
        self, unit_df, h5_path, output_csv_path, checkpoint=None
    ):
        """
        Extract features using a Hugging Face computer vision model for all units and save them.

        This method loads a pretrained computer vision model, iterates through all
        spectrograms in the HDF5 dataset, encodes them into fixed-length
        embeddings, and writes a combined DataFrame containing both
        metadata and features to CSV. The features are stored in columns
        named 'cv_feat_{i}'.

        Parameters
        ----------
        unit_df : pd.DataFrame
            DataFrame containing unit metadata with a column 'h5_index'
            referring to indices in the HDF5 'spectrograms' dataset.
        h5_path : str or Path
            Path to the HDF5 file containing spectrograms.
        output_csv_path : str or Path
            Path to the CSV file in which to store metadata and
            features.
        checkpoint : str, optional
            Hugging Face model checkpoint name. If None, a default
            checkpoint from self.config['vision_checkpoint'] is used,
            or 'facebook/dinov3-vitb16-pretrain-lvd1689m' if that key
            is not present. The default is None.

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing metadata and features for all units,
            or None if the model could not be loaded successfully.
        """
        print("\n--- Starting computer vision feature extraction ---")

        # Ensure we have data to process
        if "h5_index" not in unit_df.columns:
            print("Error: unit_df must contain an 'h5_index' column")
            return None

        # Sort by h5_index to enforce a stable and known order
        unit_df_sorted = unit_df.sort_values("h5_index").reset_index(drop=True)
        all_indices = unit_df_sorted["h5_index"].tolist()

        if not all_indices:
            print("No data found in the DataFrame for feature extraction")
            return None

        # Create dataset and DataLoader
        dataset = SpectrogramDataset(h5_path, all_indices)
        num_workers = min(4, os.cpu_count() or 1)
        batch_size = self.config.get("batch_size", 32) * 2

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # Select vision checkpoint
        model_name = checkpoint or self.config.get(
            "vision_checkpoint",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
        )

        # Determine vision device (auto-detect if not specified)
        vision_device = self.config.get("vision_device")
        if vision_device is None:
            if torch.backends.mps.is_available():
                vision_device = "mps"
            elif torch.cuda.is_available():
                vision_device = "cuda"
            else:
                vision_device = "cpu"

        # Load pre-trained image processor and model
        try:
            print(f"Loading vision model checkpoint: {model_name}")
            processor = AutoImageProcessor.from_pretrained(model_name)
            cv_model = AutoModel.from_pretrained(model_name)
            cv_model.to(vision_device)
            cv_model.eval()
        except Exception as e:
            print(f"Could not load vision model '{model_name}': {e}")
            return None

        # Extract features in batches
        all_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting vision features"):
                # Batch shape: (B, 1, H, W), float in [0, 1] if stored as uint8.
                # Repeat single channel to 3 channels for RGB-based processors.
                # Keep tensors on CPU for the processor.
                images = batch.repeat(1, 3, 1, 1)  # (B, 3, H, W)

                # Let the processor handle resizing and normalization
                inputs = processor(
                    images=images,
                    return_tensors="pt",
                ).to(vision_device)

                # Perform forward pass through model
                outputs = cv_model(**inputs)

                # Prefer pooler_output when available
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    features = outputs.pooler_output  # (B, D)
                # Fallback: use last_hidden_state
                elif hasattr(outputs, "last_hidden_state"):
                    last_hidden = outputs.last_hidden_state
                    # ViT-style: (B, N, D) -> use CLS token
                    if last_hidden.dim() == 3:
                        features = last_hidden[:, 0, :]
                    # ConvNeXt-style: (B, C, H, W) -> global average pool
                    elif last_hidden.dim() == 4:
                        features = last_hidden.mean(dim=(2, 3))
                    else:
                        features = last_hidden.view(last_hidden.size(0), -1)
                # Final fallback: first tensor in outputs, flattened
                else:
                    first = (
                        outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    )
                    if isinstance(first, torch.Tensor):
                        features = first.view(first.size(0), -1)
                    else:
                        raise RuntimeError(
                            "Could not infer feature tensor from model outputs; "
                            "try a different checkpoint or adapt extraction logic.",
                        )

                all_features.append(features.cpu())

        # Concatenate features and create DataFrame
        if not all_features:
            print("No vision features were extracted")
            return None

        comp_viz_features = torch.cat(all_features, dim=0).numpy()
        feature_dim = comp_viz_features.shape[1]

        # Create cv_feat columns
        cv_feature_cols = [f"cv_feat_{i}" for i in range(feature_dim)]
        cv_feature_df = pd.DataFrame(comp_viz_features, columns=cv_feature_cols)

        # Combine metadata with features
        result_df = pd.concat([unit_df_sorted, cv_feature_df], axis=1)

        # Save final CSV
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv_path, index=False)
        print(
            f"\n--- Vision feature pipeline complete. Exported data for {len(result_df)} units to {output_csv_path} ---"
        )

        return result_df
