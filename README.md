<div align="center">
<img src="https://raw.githubusercontent.com/masonyoungblood/chatter/refs/heads/main/docs/_static/logo.png" alt="chatter logo" width="300">

[Mason Youngblood](https://masonyoungblood.com/)

[![pypi](https://img.shields.io/pypi/v/chatter-pkg?color=440154)](https://pypi.org/project/chatter-pkg/)
[![python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmasonyoungblood%2Fchatter%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&color=414487)](https://www.python.org/)
[![size](https://img.shields.io/github/repo-size/masonyoungblood/chatter?color=2A788E)](https://github.com/masonyoungblood/chatter)
[![license](https://img.shields.io/badge/license-MIT-22A884)](https://github.com/masonyoungblood/chatter/blob/main/LICENSE)
[![tests](https://img.shields.io/github/actions/workflow/status/masonyoungblood/chatter/tests.yml?branch=main&color=7ad151)](https://github.com/masonyoungblood/chatter/actions)
[![doi](https://img.shields.io/badge/doi-10.TBD/arxiv.TBD-fde725)](https://arxiv.org/)

[Full Documentation](https://masonyoungblood.github.io/chatter)
</div>


# `chatter`: a Python library for applying information theory and AI/ML models to animal communication






The study of animal communication often involves categorizing units into types (e.g. syllables in songbirds, or notes in humpback whales). While this approach is useful in many cases, it necessarily flattens the complexity and nuance present in real communication systems. `chatter` is a new Python library for analyzing animal communication in continuous latent space using information theory and modern machine learning techniques. It is taxonomically agnostic, and has been tested with the vocalizations of birds, bats, whales, and primates. By leveraging a variety of different architectures, including variational autoencoders and vision transformers, `chatter` represents vocal sequences as trajectories in high-dimensional latent space, bypassing the need for manual or automatic categorization of units. The library provides an end-to-end workflow—from preprocessing and segmentation to model training and feature extraction—that enables researchers to quantify features like:

- Complexity: path length of sequences in latent space per unit time.
- Predictability: predictability of a transition in latent space.
- Similarity: cosine similarity between units or sequences in latent space.
- Novelty: inverse of predicted density of units or sequences in latent space.

Below is a basic diagram of the `chatter` workflow, showing the progression from spectrograms to latent features to visualizations in 2D space.

![workflow](docs/_images/diagram.png)

Additionally, `chatter` makes it easy to explore the latent space of a species' vocalizations, either statically or with an interactive plot like the one below (of syllables in Cassin's vireo song).

![embeddings](docs/_images/cassins_vireo_embedding.gif)

This project is heavily inspired by the work of folks like Nilo Merino Recalde and Tim Sainburg. Here is a list of related projects:

- Sainburg, T., Thielk, M., Gentner, T. Q. (2020). Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires. *PLOS Computational Biology*. [https://doi.org/10.1371/journal.pcbi.1008228](https://doi.org/10.1371/journal.pcbi.1008228)
- Goffinet, J., Brudner, S., Mooney, R., Pearson, J. (2021). Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires. *eLife*. [https://doi.org/10.7554/eLife.67855](https://doi.org/10.7554/eLife.67855)
- Merino Recalde, N. (2023). pykanto: a python library to accelerate research on wild bird song. *Methods in Ecology and Evolution*. [https://doi.org/10.1111/2041-210X.14155](https://doi.org/10.1111/2041-210X.14155)
- Alam, D., Zia, F., Roberts, T. F. (2024). The hidden fitness of the male zebra finch courtship song. *Nature*. [https://www.doi.org/10.1038/s41586-024-07207-4](https://www.doi.org/10.1038/s41586-024-07207-4)

Please cite `chatter` as:

- Youngblood, M. (2025). Chatter: a Python library for applying information theory and AI/ML models to animal communication (v0.1.4). *GitHub*. [https://github.com/masonyoungblood/chatter](https://github.com/masonyoungblood/chatter)

```bibtex
@software{youngblood_chatter_2025,
   author = {Youngblood, Mason},
   title = {Chatter: a Python library for applying information theory and AI/ML models to animal communication},
   version = {v0.1.4},
   date = {2025},
   publisher = {GitHub},
   url = {https://github.com/masonyoungblood/chatter}
}
```

# Installing `chatter`

`chatter` should always be installed inside a new virtual environment. To create an environment using `conda` you can run:

```bash
conda create -n chatter python==3.13.3
```

Then, you can activate the environment and install from GitHub using `pip` or `uv`:

```bash
conda activate chatter
pip install chatter-pkg
```

```bash
conda activate chatter
uv pip install chatter-pkg
```

Note that `chatter` uses `torch` as its machine learning backend, and was developed to use GPU acceleration on Apple Silicon. If you run into issues with compatibility, please look into the `torch` [documentation](https://docs.pytorch.org/docs/main/index.html) before opening an issue on GitHub.
