import re

# Read configuration from build script
config = {}
with open("patch_config.txt", "r") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            config[key] = value

DATE = config.get("DATE", "")
GITHUB_URL = config.get("GITHUB_URL", "")
DOCUMENTATION_URL = config.get("DOCUMENTATION_URL", "")
PYPI_URL = config.get("PYPI_URL", "")
CITATION = config.get("CITATION", "")
LOGO_WIDTH = config.get("LOGO_WIDTH", "3cm")

with open("paper.tex", "r") as f:
    content = f.read()

# ============================================================================
# Standard JOSS template fixes (biblatex removal, etc.)
# ============================================================================

# Remove biblatex package import
content = re.sub(r"\\usepackage\[.*?\]\{biblatex\}", "", content, flags=re.DOTALL)

# Remove bibliography resource definitions
content = re.sub(r"\\bibliography\{.*?\}", "", content)
content = re.sub(r"\\addbibresource\{.*?\}", "", content)

# Remove printbibliography command
content = re.sub(r"\\printbibliography\[.*?\]", "", content)
content = re.sub(r"\\printbibliography", "", content)

# Replace \renewcommand{\bibfont} with \newcommand since biblatex isn't loaded
content = re.sub(r"\\renewcommand\{\\bibfont\}", r"\\newcommand{\\bibfont}", content)

# ============================================================================
# Preprint customizations
# ============================================================================

# 1. Replace the Software itemize section - swap Review/Repository/Archive with GitHub/PyPI
old_items = r"""\\begin\{itemize\}
    \\setlength\\itemsep\{0em\}
    \\item \\href\{[^}]*\}\{\\color\{linky\}\{Review\}\} \\ExternalLink
    \\item \\href\{[^}]*\}\{\\color\{linky\}\{Repository\}\} \\ExternalLink
    \\item \\href\{[^}]*\}\{\\color\{linky\}\{Archive\}\} \\ExternalLink
  \\end\{itemize\}"""

new_items = f"""\\\\begin{{itemize}}
    \\\\setlength\\\\itemsep{{0em}}
    \\\\item \\\\href{{{GITHUB_URL}}}{{\\\\color{{linky}}{{GitHub}}}} \\\\ExternalLink
    \\\\item \\\\href{{{DOCUMENTATION_URL}}}{{\\\\color{{linky}}{{Documentation}}}} \\\\ExternalLink
    \\\\item \\\\href{{{PYPI_URL}}}{{\\\\color{{linky}}{{PyPI}}}} \\\\ExternalLink
  \\\\end{{itemize}}"""

content = re.sub(old_items, new_items, content)

# 2. Remove DOI field
content = re.sub(r"\{\\bfseries DOI:\}[^\n]*\n", "", content)

# 3. Replace Submitted/Published with just Preprint
content = re.sub(r"\{\\bfseries Submitted:\}[^\n]*\\\\", "", content)
content = re.sub(
    r"\{\\bfseries Published:\}[^\n]*", f"{{\\\\bfseries Date:}} {DATE}", content
)

# 4. Remove the entire License section
content = re.sub(r"\{\\bfseries License\}\\\\[^}]+CC BY 4\.0[^}]*\}\)\.", "", content)
content = re.sub(
    r"\\vspace\{2mm\}\s*\n\s*\{\\bfseries License\}.*?CC BY 4\.0.*?\)\.",
    "",
    content,
    flags=re.DOTALL,
)

# 5. Adjust logo width
content = re.sub(
    r"\\includegraphics\[width=[^\]]+\](\{[^}]*logo[^}]*\})",
    f"\\\\includegraphics[width={LOGO_WIDTH}]\\1",
    content,
)

# 6. Replace the footer with custom citation
# Find and replace the \fancyfoot[L] command with nested braces
# This regex matches \fancyfoot[L]{ followed by content with balanced braces }
if CITATION:
    escaped_citation = (
        CITATION.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
    )

    # Match the entire \fancyfoot[L]{...} with nested braces using a function
    def replace_fancyfoot(match):
        return f"\\fancyfoot[L]{{\\parbox[t]{{0.98\\headwidth}}{{\\footnotesize{{\\sffamily {escaped_citation}}}}}}}"

    # Find \fancyfoot[L]{ and match until we find the balanced closing brace
    # Simple approach: find \fancyfoot[L]{ and replace up to end of line, handling multiline
    lines = content.split("\n")
    new_lines = []
    skip_until_brace_close = False
    brace_count = 0

    for line in lines:
        if skip_until_brace_close:
            # Count braces to find the end
            for char in line:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
            if brace_count <= 0:
                skip_until_brace_close = False
            continue

        if "\\fancyfoot[L]" in line:
            # Start of the footer command - replace with our version
            new_lines.append(
                f"\\fancyfoot[L]{{\\parbox[t]{{0.98\\headwidth}}{{\\footnotesize{{\\sffamily {escaped_citation}}}}}}}"
            )
            # Check if the command continues on next lines
            brace_count = line.count("{") - line.count("}")
            if brace_count > 0:
                skip_until_brace_close = True
        else:
            new_lines.append(line)

    content = "\n".join(new_lines)

# ============================================================================
# Add compatibility definitions after \documentclass
# ============================================================================

preamble = """
% Pandoc citeproc compatibility
\\providecommand{\\pandocbounded}[1]{#1}

% Make \\bibitem work outside thebibliography environment
\\makeatletter
\\newcommand{\\citeproctext}{}
\\newcounter{citeproccounter}
\\let\\olditem\\item
\\renewcommand{\\item}[1][]{\\olditem}
\\makeatother
"""

lines = content.split("\n")
new_lines = []
for line in lines:
    new_lines.append(line)
    if line.strip().startswith("\\documentclass"):
        new_lines.append(preamble)

content = "\n".join(new_lines)

# Replace \bibitem with simple paragraph breaks for CSL references
content = re.sub(
    r"\\bibitem\[\\citeproctext\]\{[^}]+\}\s*\n", r"\\par\\noindent ", content
)

with open("paper.tex", "w") as f:
    f.write(content)

print("LaTeX file patched successfully.")
print(f"  - Date: {DATE}")
print(f"  - GitHub: {GITHUB_URL}")
print(f"  - PyPI: {PYPI_URL}")
print(f"  - Logo width: {LOGO_WIDTH}")
print(
    f"  - Citation: {CITATION[:50]}..."
    if len(CITATION) > 50
    else f"  - Citation: {CITATION}"
)
