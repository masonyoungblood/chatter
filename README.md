# `chatter`: a Python library for applying information theory and AI/ML models to animal communication

[Mason Youngblood](https://masonyoungblood.com/)

![python](https://img.shields.io/badge/_python-3.13+-440154) ![version](https://img.shields.io/badge/_version-0.1.0-21918c) ![doi](https://img.shields.io/badge/_doi-TBD-fde725)

<br><br>
**[Full Documentation](https://masonyoungblood.github.io/chatter/docs/_build/html/index.html)**

Historically, analyses of sequential structure in animal communication have involved the identification of unit types (e.g. "syllables" in bird song and "notes" in whale song). This collapses continuous variation into discrete categories that align with human perception, a process that loses a great deal of the complexity and nuance present in the actual signals. Recent innovations in machine learning, such as variational autoencoders and vision transformers, allow us to bypass discretization and analyze animal communication signals directly in continuous space. `chatter` makes it easy for researchers to apply these methods to their data, to quantify features like:

- Complexity—path length of sequences in latent space per unit time.
- Predictability—predictability of a transition in latent space.
- Similarity—cosine similarity between units or sequences in latent space.
- Novelty—inverse of predicted density of units or sequences in latent space.

Additionally, `chatter` makes it easy to explore the latent space of a species' vocalizations, either statically or with an interactive plot like the one below (of syllables in Cassin's vireo song).

![embeddings](docs/_static/cassins_vireo_embedding.gif)

This project is heavily inspired by the work of folks like Nilo Merino Recalde and Tim Sainburg. Here is a list of related projects:

- Sainburg, T., Thielk, M., Gentner, T. Q. (2020). Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires. *PLOS Computational Biology*. [https://doi.org/10.1371/journal.pcbi.1008228](https://doi.org/10.1371/journal.pcbi.1008228)
- Goffinet, J., Brudner, S., Mooney, R., Pearson, J. (2021). Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires. *eLife*. [https://doi.org/10.7554/eLife.67855](https://doi.org/10.7554/eLife.67855)
- Merino Recalde, N. (2023). pykanto: a python library to accelerate research on wild bird song. *Methods in Ecology and Evolution*. [https://doi.org/10.1111/2041-210X.14155](https://doi.org/10.1111/2041-210X.14155)
- Alam, D., Zia, F., Roberts, T. F. (2024). The hidden fitness of the male zebra finch courtship song. *Nature*. [https://www.doi.org/10.1038/s41586-024-07207-4](https://www.doi.org/10.1038/s41586-024-07207-4)

Please cite `chatter` as:

{{ apa_citation }}

{{ bibtex_citation }}

# Installing `chatter`

`chatter` should always be installed inside a new virtual environment. To create an environment using `conda` you can run:

```bash
conda create -n chatter python==3.13.3
```

Then, you can activate the environment and install from GitHub using `pip`:

```bash
conda activate chatter
pip install git+https://github.com/masonyoungblood/chatter.git
```

Note that `chatter` uses `torch` as its machine learning backend, and was developed to use GPU acceleration on Apple Silicon. If you run into issues with compatibility, please look into the `torch` [documentation](https://docs.pytorch.org/docs/main/index.html) before opening an issue on GitHub.
