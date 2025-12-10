import os
import sys


def generate_readme():
    # Paths are relative to this script location (docs/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    # Add docs dir to path to import conf.py
    sys.path.append(base_dir)

    # Try to import conf to get substitutions
    substitutions = {}
    try:
        import conf

        if hasattr(conf, "myst_substitutions"):
            substitutions = conf.myst_substitutions
            print("Loaded substitutions from conf.py")
    except ImportError:
        print("Warning: Could not import conf.py. Substitutions will not be applied.")
    except Exception as e:
        print(
            f"Warning: Error importing conf.py: {e}. Substitutions will not be applied."
        )

    index_path = os.path.join(base_dir, "index.md")
    install_path = os.path.join(base_dir, "contents", "installation.md")
    readme_path = os.path.join(project_root, "README.md")

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

    # Modifications requested:
    # 1. Drop "Home" from top
    # 2. Change chatter title to header (#)
    # 3. Move doc link to after <br><br>

    # 1. Remove "# Home"
    index_content = index_content.replace("# Home", "")

    # 2. Convert title line to H1
    old_title = '**<span style="font-size:larger;">`chatter`: a Python library for applying information theory and AI/ML models to animal communication</span>**'
    new_title = "# `chatter`: a Python library for applying information theory and AI/ML models to animal communication"
    index_content = index_content.replace(old_title, new_title)

    # 3. Insert doc link after <br><br>
    doc_link = "**[Full Documentation](https://masonyoungblood.github.io/chatter/docs/_build/html/index.html)**"
    br_marker = "<br><br>"

    if br_marker in index_content:
        index_content = index_content.replace(br_marker, f"{br_marker}\n\n{doc_link}")
    else:
        # Fallback if marker not found
        print("Warning: <br><br> marker not found. Appending doc link to top.")
        index_content = doc_link + "\n\n" + index_content

    # Fix relative image paths
    # Replace _static/ with docs/_static/
    index_content = index_content.replace("(_static/", "(docs/_static/")

    # Apply substitutions
    for key, value in substitutions.items():
        placeholder = f"{{{{ {key} }}}}"
        if placeholder in index_content:
            index_content = index_content.replace(placeholder, str(value))
        if placeholder in install_content:
            install_content = install_content.replace(placeholder, str(value))

    # Clean up multiple newlines that might result from removing "# Home"
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
