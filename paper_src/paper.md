---
title: '`chatter`: a Python library for applying information theory and AI/ML models to animal communication'
authors:
  - name: Mason Youngblood
affiliations:
  - name: Institute for Advanced Computational Science, Stony Brook University, USA        masonyoungblood@gmail.com
date: 12 December 2025
bibliography: paper.bib
logo: logo.png
logo_width: 5cm
github_url: https://github.com/masonyoungblood/chatter
documentation_url: https://masonyoungblood.github.io/chatter
pypi_url: https://pypi.org/project/chatter-pkg/
citation: "Youngblood, M. (2025). \textit{chatter}: a Python library for applying information theory and AI/ML models to animal communication."

---

# Summary

The study of animal communication often involves categorizing units into types (e.g. syllables in songbirds, or notes in humpback whales). While this approach is useful in many cases, it necessarily flattens the complexity and nuance present in real communication systems. `chatter` is a new Python library for analyzing animal communication in continuous latent space using information theory and modern machine learning techniques. It is taxonomically agnostic, and has been tested with the vocalizations of birds, bats, whales, and primates. By leveraging a variety of different architectures, including variational autoencoders and vision transformers, `chatter` represents vocal sequences as trajectories in high-dimensional latent space, bypassing the need for manual or automatic categorization of units. The library provides an end-to-end workflow—from preprocessing and segmentation to model training and feature extraction—that enables researchers to quantify the complexity, predictability, similarity, and novelty of vocal sequences.

# Statement of Need

In recent years, animal behaviorists have started to use machine learning to project vocalizations into high-dimensional latent space [@sainburg2020; @goffinet2021; @recalde2023]. However, these methods are typically used to identify categories of vocalizations instead of enabling analysis of vocal sequences as continuous vectors, with some exceptions [@alam2024]. `chatter` addresses this gap by providing an accessible, modular, and easy-to-use interface for applying these techniques. It complements existing tools like `pykanto` [@recalde2023] and `AVGN` [@sainburg2020], which focus on discrete analysis, by offering a parallel workflow for continuous analysis. In doing so, it lowers the barrier to entry for researchers who are interested in computational methods but may not have the time or expertise to assemble a machine learning pipeline from scratch. The library is designed to be flexible enough for advanced users while providing sensible defaults for novices.

The core workflow of `chatter` has three steps, all managed by a configuration dictionary. First, users initialize the `Analyzer` class, which takes parameters like minimum and maximum frequency, amplitude thresholds, and denoising settings. The `Analyzer` handles the raw audio, preprocessing it with noise reduction, filtering, and compression, and segmenting it into individual acoustic units based on amplitude or other criteria. It also gives users working with birdsong data the ability to identify their focal species using BirdNET [@kahl2021]. The resulting units are converted into spectrograms.

![A basic diagram of the `chatter` workflow, showing the progression from spectrograms to latent features to visualizations in 2D space. Note that all of the information theoretic analysis occurs in original latent space, not in the reduced 2D space.](diagram.pdf)

Next, the `Trainer` class is used to train a convolutional variational autoencoder to learn a high-dimensional latent representation of the vocalizations [@goffinet2021]. This unsupervised approach is able to describe complex vocalizations in a relatively lossless way—allowing users to reconstruct the original sound from a set of learned acoustic features. These latent features are then extracted for all acoustic units. Users also have the option of collecting latent features by applying the DINOv3 vision transformer to spectrograms, which works especially well when treating sequences as units (i.e. working with spectrograms of whole sequences rather than units within sequences) [@simeoni2025]. Finally, the library provides tools to visualize this latent space using dimensionality reduction techniques such as PaCMAP [@wang2021], an extension of UMAP that better captures global structure. Figure 2 illustrates this output for Cassin's vireo syllables (recordings from @hedley2016).

![The latent space of Cassin's vireo syllables. The plot visualizes the syllables in a 2D latent space produced by applying PaCMAP to the latent features from a variational autoencoder, with representative spectrograms overlaid.](cassins_vireo_embedding.png)

Finally, the `FeatureProcessor` class provides a variety of functions for downstream analysis. Complexity is quantified by calculating the path length (cosine distance) between consecutive acoustic units in latent space, normalized by duration, offering a continuous measure of acoustic diversity over time [@alam2024] similar to previous methods based on spectrogram cross-correlation [@sawant2022]. To assess predictability, `chatter` trains a vector autoregression model on sequences of latent features [@sims1980]. This model learns temporal dependencies and quantifies the probability of future states. `chatter` also estimates the commonness and rarity of acoustic units using `denmarf`, a density estimation library that uses masked autoregressive flows to compute density in high-dimensional space [@papamakarios2017]. This provides a measure of how common or rare a given vocalization is relative to the learned distribution. Finally, similarity between whole sequences of units is computed by applying dynamic time warping to the sequences of latent vectors (i.e. panel B in Figure 1). Collectively, these measures provide a fairly complex and nuanced description of animal vocal sequences.

# References
