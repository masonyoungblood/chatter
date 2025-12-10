import os
import sys
import tomllib


def _load_version(pyproject_path: str) -> str:
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return str(data["project"]["version"])
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
        return "0.0.0"


def generate_readme():
    # Paths are relative to this script location (docs/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    index_path = os.path.join(base_dir, "index.md")
    install_path = os.path.join(base_dir, "contents", "installation.md")
    readme_path = os.path.join(project_root, "README.md")
    pyproject_path = os.path.join(project_root, "pyproject.toml")

    try:
        with open(index_path, "r") as f:
            index_content = f.read()

        with open(install_path, "r") as f:
            install_content = f.read()
    except FileNotFoundError as e:
        print(f"Error reading source files: {e}")
        return

    # Remove TOC from index.md
    split_marker = '<div id="main-page">'
    if split_marker in index_content:
        index_content = index_content.split(split_marker)[0]
    else:
        print("Warning: Split marker not found in index.md")

    # Drop "Home" header
    index_content = index_content.replace("# Home", "")

    # Convert title line to H1
    old_title = '**<span style="font-size:larger;">`chatter`: a Python library for applying information theory and AI/ML models to animal communication</span>**'
    new_title = "# `chatter`: a Python library for applying information theory and AI/ML models to animal communication"
    index_content = index_content.replace(old_title, new_title)

    # Remove author/badges/doc link placeholders from body; we'll inject a centered block
    index_content = index_content.replace(
        "[Mason Youngblood](https://masonyoungblood.com/)", ""
    )
    index_content = index_content.replace(
        "{{ python_badge }} {{ version_badge }} {{ doi_badge }}", ""
    )
    index_content = index_content.replace("{{ python_badge }}", "")
    index_content = index_content.replace("{{ version_badge }}", "")
    index_content = index_content.replace("{{ doi_badge }}", "")
    index_content = index_content.replace(
        "**[Full Documentation](https://masonyoungblood.github.io/chatter/docs/_build/html/index.html)**",
        "",
    )
    index_content = index_content.replace("<br><br>", "")

    # Fix relative image paths - use GitHub raw URLs for README
    index_content = index_content.replace(
        "(_static/cassins_vireo_embedding.gif)",
        "(https://raw.githubusercontent.com/masonyoungblood/chatter/main/docs/_static/cassins_vireo_embedding.gif)",
    )

    # Build substitutions locally (avoid heavy imports)
    version = _load_version(pyproject_path)
    # Prefer the declared requires-python for the badge to avoid env/version drift.
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        py_req = str(data["project"].get("requires-python", "")).strip()
    except Exception:
        py_req = ""

    if py_req.startswith(">="):
        py_badge_val = py_req.replace(">=", "").strip() + "+"
    elif py_req:
        py_badge_val = py_req
    else:
        py_badge_val = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    py_ver = py_badge_val
    current_year = str(__import__("datetime").datetime.now().year)
    # Compose citations without importing the package
    apa_citation = (
        f"- Youngblood, M. ({current_year}). "
        f"Chatter: a Python library for applying information theory and AI/ML models to animal communication (v{version}). "
        f"*GitHub*. [https://github.com/masonyoungblood/chatter](https://github.com/masonyoungblood/chatter)"
    )
    bibtex_citation = (
        "```bibtex\n"
        f"@software{{youngblood_chatter_{current_year},\n"
        "   author = {Youngblood, Mason},\n"
        "   title = {Chatter: a Python library for applying information theory and AI/ML models to animal communication},\n"
        f"   version = {{v{version}}},\n"
        f"   date = {{{current_year}}},\n"
        "   publisher = {GitHub},\n"
        "   url = {https://github.com/masonyoungblood/chatter}\n"
        "}\n"
        "```"
    )
    substitutions = {
        "python_badge": f"![python](https://img.shields.io/badge/_python-{py_ver}-440154)",
        "version_badge": f"![version](https://img.shields.io/badge/_version-{version}-21918c)",
        "doi_badge": "![doi](https://img.shields.io/badge/_doi-TBD-fde725)",
        "python_version": py_ver,
        "chatter_version": version,
        "apa_citation": apa_citation,
        "bibtex_citation": bibtex_citation,
    }

    # Build centered header block with logo, name, and doc link
    center_block = (
        '<div align="center">\n'
        '<img src="https://raw.githubusercontent.com/masonyoungblood/chatter/main/docs/_static/logo.png" alt="chatter logo" width="400">\n\n'
        "[Mason Youngblood](https://masonyoungblood.com/)\n\n"
        "**[Full Documentation](https://masonyoungblood.github.io/chatter/docs/_build/html/index.html)**\n"
        "</div>\n"
    )

    # Inject centered block after the first line (title)
    if "\n" in index_content:
        first_line, rest = index_content.split("\n", 1)
        index_content = first_line + "\n\n" + center_block + "\n" + rest
    else:
        index_content = index_content + "\n\n" + center_block

    # Apply substitutions
    for key, value in substitutions.items():
        placeholder = f"{{{{ {key} }}}}"
        if placeholder in index_content:
            index_content = index_content.replace(placeholder, str(value))
        if placeholder in install_content:
            install_content = install_content.replace(placeholder, str(value))

    index_content = index_content.strip()
    final_content = index_content + "\n\n" + install_content

    try:
        with open(readme_path, "w") as f:
            f.write(final_content)
        print(f"Successfully generated {readme_path}")
    except Exception as e:
        print(f"Error writing README.md: {e}")


if __name__ == "__main__":
    generate_readme()
