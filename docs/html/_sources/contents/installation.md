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
