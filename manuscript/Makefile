# Makefile for building LaTeX manuscript
# Usage:
#   make        - Full build (pdflatex -> biber -> pdflatex x2)
#   make quick  - Single pdflatex pass (no bibliography)
#   make clean  - Remove all build artifacts

.PHONY: all quick clean

OUTDIR = out
SRC = main.tex
STEM = $(basename $(SRC))
TEX = pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=$(OUTDIR)

ifeq ($(OS),Windows_NT)
    DEVNULL = NUL
else
    DEVNULL = /dev/null
endif

all:
	@echo [1/4] pdflatex (initial)
	@$(TEX) $(SRC) > $(DEVNULL)
	@echo [2/4] biber
	@biber --output-directory=$(OUTDIR) $(STEM) > $(DEVNULL)
	@echo [3/4] pdflatex (resolve references)
	@$(TEX) $(SRC) > $(DEVNULL)
	@echo [4/4] pdflatex (final)
	@$(TEX) $(SRC) > $(DEVNULL)

quick:
	@echo [1/1] pdflatex
	@$(TEX) $(SRC) > $(DEVNULL)

clean:
	rm -rf $(OUTDIR)/*