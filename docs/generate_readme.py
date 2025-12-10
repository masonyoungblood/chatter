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

    # Insert doc link after <br><br> with a single trailing newline
    doc_link = "**[Full Documentation](https://masonyoungblood.github.io/chatter/docs/_build/html/index.html)**"
    br_marker = "<br><br>"

    if br_marker in index_content:
        index_content = index_content.replace(br_marker, f"{br_marker}\n{doc_link}")
    else:
        # Fallback if marker not found
        print("Warning: <br><br> marker not found. Appending doc link to top.")
        index_content = doc_link + "\n" + index_content

    # Fix relative image paths
    index_content = index_content.replace("(_static/", "(docs/_static/")

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
    substitutions = {
        "python_badge": f"![python](https://img.shields.io/badge/_python-{py_ver}-440154)",
        "version_badge": f"![version](https://img.shields.io/badge/_version-{version}-21918c)",
        "doi_badge": "![doi](https://img.shields.io/badge/_doi-TBD-fde725)",
        "python_version": py_ver,
        "chatter_version": version,
    }

    # Make badges inline if they appear stacked
    stacked = "{{ python_badge }}\n{{ version_badge }}\n{{ doi_badge }}"
    inline = "{{ python_badge }} {{ version_badge }} {{ doi_badge }}"
    index_content = index_content.replace(stacked, inline)

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
