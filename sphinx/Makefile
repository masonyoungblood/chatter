# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DOCSDIR       = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html deploy

# Clean build directory
clean:
	@rm -rf $(BUILDDIR)
	@echo "Build directory cleaned."

# Build HTML documentation
html: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@python generate_readme.py
	@echo "HTML documentation built in $(BUILDDIR)/html/"

# Deploy: build and copy to /docs for GitHub Pages
deploy: html
	@echo "Deploying to $(DOCSDIR) for GitHub Pages..."
	@rm -rf $(DOCSDIR)/*
	@cp -r $(BUILDDIR)/html/* $(DOCSDIR)/
	@touch $(DOCSDIR)/.nojekyll
	@echo "Documentation deployed to $(DOCSDIR)/"
	@echo "Commit and push $(DOCSDIR) to publish to GitHub Pages."

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
