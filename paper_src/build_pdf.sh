#!/bin/bash

# This script should be run from the paper_src/ directory

# Clean up old generated files (keep paper.md and paper.bib)
echo "Cleaning up old generated files..."
rm -f paper_temp.md
rm -f paper.aux
rm -f paper.fdb_latexmk
rm -f paper.fls
rm -f paper.log
rm -f paper.out
rm -f paper.pdf
rm -f paper.tex
rm -f paper.bbl
rm -f paper.bcf
rm -f paper.blg
rm -f paper.run.xml
rm -f patch_config.txt
rm -f joss.latex
rm -f apa.csl

# JOSS resources URLs
TEMPLATE_URL="https://raw.githubusercontent.com/openjournals/whedon/master/resources/latex.template"
CSL_URL="https://raw.githubusercontent.com/openjournals/whedon/master/resources/apa.csl"

echo "Downloading JOSS resources..."
curl -s -L -o "joss.latex" "$TEMPLATE_URL"
curl -s -L -o "apa.csl" "$CSL_URL"

# Extract custom fields from YAML header
DATE=$(grep "^date:" paper.md | sed 's/date: *//')
GITHUB_URL=$(grep "^github_url:" paper.md | sed 's/github_url: *//')
DOCUMENTATION_URL=$(grep "^documentation_url:" paper.md | sed 's/documentation_url: *//')
PYPI_URL=$(grep "^pypi_url:" paper.md | sed 's/pypi_url: *//')
CITATION=$(grep "^citation:" paper.md | sed 's/citation: *"//' | sed 's/"$//')
LOGO=$(grep "^logo:" paper.md | sed 's/logo: *//')
LOGO_WIDTH=$(grep "^logo_width:" paper.md | sed 's/logo_width: *//')

# Default logo width if not specified
if [ -z "$LOGO_WIDTH" ]; then
    LOGO_WIDTH="3cm"
fi

# Use default placeholder if no logo specified
if [ -z "$LOGO" ]; then
    LOGO="placeholder-logo.png"
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP6DwABBAEBO0YY2QAAAABJRU5ErkJggg==" | base64 -d > "$LOGO"
    CREATED_PLACEHOLDER=true
else
    echo "Using custom logo: $LOGO"
    CREATED_PLACEHOLDER=false
fi

# Write config for patch_tex.py
cat > patch_config.txt << EOF
DATE=$DATE
GITHUB_URL=$GITHUB_URL
DOCUMENTATION_URL=$DOCUMENTATION_URL
PYPI_URL=$PYPI_URL
CITATION=$CITATION
LOGO_WIDTH=$LOGO_WIDTH
EOF

echo "Preparing Markdown..."
# Remove bibliography and custom fields from YAML to prevent Pandoc issues
sed '/^bibliography: paper.bib/d; /^github_url:/d; /^documentation_url:/d; /^pypi_url:/d; /^citation:/d; /^logo:/d; /^logo_width:/d' "paper.md" > "paper_temp.md"

echo "Generating LaTeX file using local Pandoc..."
# Generate .tex with embedded citations via citeproc
pandoc "paper_temp.md" \
  -o "paper.tex" \
  --template="joss.latex" \
  --csl="apa.csl" \
  --bibliography="paper.bib" \
  --citeproc \
  --variable logo_path="$LOGO" \
  --variable journal="Journal of Open Source Software" \
  --variable year="2025" \
  --variable submitted="$DATE" \
  --variable published="$DATE" \
  --variable issue="1" \
  --variable volume="10" \
  --variable page="1" \
  --variable doi="10.21105/joss.00000" \
  --variable citation_author="Youngblood" \
  --variable paper_title="Chatter: a Python library for applying information theory and AI/ML models to animal communication" \
  --variable graphics="true" \
  --variable geometry="margin=1in"

echo "Patching generated LaTeX file using Python..."
python3 patch_tex.py

echo "Compiling PDF using Dockerized LaTeX..."
docker run --rm \
  --volume "$PWD:/work" \
  --workdir /work \
  texlive/texlive \
  latexmk -pdf -interaction=nonstopmode -f paper.tex || true

# Check if PDF was created
if [ -f "paper.pdf" ]; then
    echo "PDF generated successfully!"

    # Create preprint copy with custom naming
    YEAR=$(date +%Y)
    mkdir -p ../paper
    cp paper.pdf "../paper/youngblood_${YEAR}_chatter.pdf"
    echo "Preprint copy saved to ../paper/youngblood_${YEAR}_chatter.pdf"
else
    echo "ERROR: PDF was not generated"
    exit 1
fi

# Note: Cleanup is now done at the beginning of the script

# Only delete placeholder logo, not user's custom logo
if [ "$CREATED_PLACEHOLDER" = true ]; then
    rm -f "$LOGO"
fi

echo "Done! PDF saved to paper.pdf"
