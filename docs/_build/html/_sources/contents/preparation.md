# Preparation

As input, `chatter` simply takes a folder of raw WAV files, which have not yet been denoised, filtered, or normalized. The input folder can have recursive structure, if you, for example, have separate folders from files from different years, locations, or individuals.

`chatter` also requires a dictionary of configuration parameters that control the analysis pipeline.

**Spectrogram Parameters**
| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `sr` | Sample rate for audio processing. | `44100` |
| `n_fft` | FFT window size. | `2048` |
| `win_length` | Window length for FFT. | `1024` |
| `hop_length` | Hop length between FFT windows. | `128` |
| `n_mels` | Number of Mel bands to generate. | `224` |
| `fmin` | Minimum frequency for Mel spectrogram. | `1000` |
| `fmax` | Maximum frequency for Mel spectrogram. | `10000` |
| `target_shape` | The dimensions spectrograms are resized to. | `(128, 128)` |

**Preprocessing Parameters**
| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `high_pass` | High-pass filter cutoff frequency in Hz. If `None`, no high-pass filter is applied. | `None` |
| `low_pass` | Low-pass filter cutoff frequency in Hz. If `None`, no low-pass filter is applied. | `None` |
| `target_dbfs`| Target dBFS for normalization. | `-20` |
| `threshold` | Noise reduction strength parameter. | `1` |
| `compressor_amount` | Amount of dynamic range compression to apply (dB). | `-20` |
| `limiter_amount` | Output limiter threshold (dB). | `-10` |
| `static` | If `True`, use a static noise profile for denoising. | `True` |
| `fade_ms` | Length of fade-in and fade-out applied to each clip (milliseconds). | `20` |
| `skip_noise` | Number of seconds to skip at the start when estimating noise. | `3.0` |
| `use_biodenoising` | If `True`, apply BioDenoising before other preprocessing. | `False` |
| `biodenoising_model` | Name of the BioDenoising model to use. | `"biodenoising16k_dns48"` |
| `use_noisereduce` | If `True`, apply `noisereduce`-based denoising. | `True` |
| `noise_floor` | Optional global noise floor for preprocessing (dBFS). If `None`, it is estimated automatically. | `None` |

**Pykanto Segmentation** (recommended for recordings of variable quality)
| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `pykanto_noise_floor` | Noise floor for segmentation (dB). | `-65` |
| `pykanto_top_dB` | Dynamic range above the noise floor to retain (dB). | `65` |
| `pykanto_max_dB` | Maximum decibel value relative to full scale. | `-30` |
| `pykanto_dB_delta` | Increment in decibels for thresholding. | `5` |
| `pykanto_min_silence_length`| Minimum duration of silence between units (seconds). | `0.001` |
| `pykanto_max_unit_length`| Maximum duration of a unit (seconds). | `0.4` |
| `pykanto_min_unit_length`| Minimum duration of a unit (seconds). | `0.03` |
| `pykanto_gauss_sigma`| Gaussian sigma for image smoothing. | `3` |
| `pykanto_silence_threshold`| Silence threshold used in segmentation. | `0.2` |

**Simple Segmentation** (recommended for recordings of high quality)
| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `simple_noise_floor`| Noise floor used during segmentation (dB). | `-60` |
| `simple_silence_threshold_db`| Silence threshold for segmentation (dB). | `-40` |
| `simple_min_silence_length`| Minimum duration of silence between units (seconds). | `0.001` |
| `simple_max_unit_length`| Maximum duration of a unit (seconds). | `0.4` |
| `simple_min_unit_length`| Minimum duration of a unit (seconds). | `0.03` |

**Autoencoder Parameters**
| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `ae_type` | Encoder–decoder architecture type: `'convolutional'` or `'vector'`. Both are implemented as variational autoencoders. | `"convolutional"` |
| `latent_dim` | Number of latent dimensions. | `32` |
| `batch_size` | Batch size for training. | `32` |
| `epochs` | Number of training epochs. | `100` |
| `lr` | Learning rate for the optimizer. | `1e-4` |
| `beta` | VAE beta parameter for balancing reconstruction and latent space smoothness. | `0.5` |

**Other Parameters**
| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `seq_bound` | Time in seconds to define sequence boundaries. | `1.0` |
| `lag_size` | Lag size for the Vector Autoregression (VAR) model. | `3` |
| `dark_mode` | If `True`, plots will use a dark theme. | `True` |
| `font` | Font used for plot labels and titles. | `'Courier'` |
| `plot_clip_duration` | Length of audio clip (seconds) used in quick demonstration plots. | `5.0` |
| `vision_checkpoint` | Name of the vision backbone checkpoint used for embeddings. | `"facebook/dinov2-base"` |
| `vision_device` | Device used for vision backbone (`"mps"`, `"cuda"`, or `"cpu"`). If `None`, it is auto-detected. | `None` |

Here is what a configuration dictionary looks like in practice. You do **not** need to specify every parameter; start from the defaults in `chatter.config` and override only what you need for your dataset.

```python
#start from defaults and override a few key parameters
user_config = {
    #spectrogram parameters
    "fmin": 1700,
    "fmax": 6000,

    #preprocessing parameters
    "high_pass": 1700,
    "low_pass": 6000,

    #simple segmentation parameters
    "simple_noise_floor": -60,

    #pykanto segmentation parameters
    "pykanto_noise_floor": -65,

    #autoencoder parameters
    "ae_type": "convolutional",
    "latent_dim": 32,
}
config = chatter.make_config(user_config)
```

I highly recommend that you experiment with the preprocessing and segmentation parameters for your particular use case. The spectrogram, autoencoder, and other parameters should work well across a wide range of species.
