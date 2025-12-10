# Home

**<span style="font-size:larger;">`chatter`: a Python library for applying information theory and AI/ML models to animal communication</span>**

[Mason Youngblood](https://masonyoungblood.com/)

<br><br>

Historically, analyses of sequential structure in animal communication have involved the identification of unit types (e.g. "syllables" in bird song and "notes" in whale song). This collapses continuous variation into discrete categories that align with human perception, a process that loses a great deal of the complexity and nuance present in the actual signals. Recent innovations in machine learning, such as variational autoencoders and vision transformers, allow us to bypass discretization and analyze animal communication signals directly in continuous space. `chatter` makes it easy for researchers to apply these methods to their data, to quantify features like:

- Complexity—path length of sequences in latent space per unit time.
- Predictability—predictability of a transition in latent space.
- Similarity—cosine similarity between units or sequences in latent space.
- Novelty—inverse of predicted density of units or sequences in latent space.

Additionally, `chatter` makes it easy to explore the latent space of a species' vocalizations, either statically or with an interactive plot like the one below (of syllables in Cassin's vireo song).

![embeddings](_static/cassins_vireo_embedding.gif)

This project is heavily inspired by the work of folks like Nilo Merino Recalde and Tim Sainburg. Here is a list of related projects:

- Sainburg, T., Thielk, M., Gentner, T. Q. (2020). Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires. *PLOS Computational Biology*. [https://doi.org/10.1371/journal.pcbi.1008228](https://doi.org/10.1371/journal.pcbi.1008228)
- Goffinet, J., Brudner, S., Mooney, R., Pearson, J. (2021). Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires. *eLife*. [https://doi.org/10.7554/eLife.67855](https://doi.org/10.7554/eLife.67855)
- Merino Recalde, N. (2023). pykanto: a python library to accelerate research on wild bird song. *Methods in Ecology and Evolution*. [https://doi.org/10.1111/2041-210X.14155](https://doi.org/10.1111/2041-210X.14155)
- Alam, D., Zia, F., Roberts, T. F. (2024). The hidden fitness of the male zebra finch courtship song. *Nature*. [https://www.doi.org/10.1038/s41586-024-07207-4](https://www.doi.org/10.1038/s41586-024-07207-4)

Please cite `chatter` as:

{{ apa_citation }}

{{ bibtex_citation }}

<div id="main-page">

```{eval-rst}
.. toctree::
   :caption: Guide
   :maxdepth: 1
   
   contents/installation
   contents/flow_diagram
   contents/preparation
```

```{eval-rst}
.. toctree::
   :caption: Full Vignettes
   :maxdepth: 1
   
   contents/cassins_vireo
```

```{eval-rst}
.. toctree::
   :caption: Short Vignettes
   :maxdepth: 1
   
   contents/humpback_whale
   contents/chimpanzee
   contents/egyptian_fruit_bat
   contents/human
```

**<span style="font-size:larger;">Reference</span>**

```{eval-rst}
.. autosummary::
   :caption: Reference
   :toctree: _autosummary
   :recursive: 
   
   chatter
```

</div>
