# Makefile for LaTeX report compilation

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Main document
DOC = report

# Output directory
BUILD_DIR = build

.PHONY: all clean view help

# Default target
all: $(DOC).pdf

# Compile the PDF (run twice for references)
$(DOC).pdf: $(DOC).tex
	@echo "Compiling LaTeX document..."
	$(LATEX) -output-directory=$(BUILD_DIR) $(DOC).tex
	@echo "Running second pass for references..."
	$(LATEX) -output-directory=$(BUILD_DIR) $(DOC).tex
	@mv $(BUILD_DIR)/$(DOC).pdf .
	@echo "PDF generated: $(DOC).pdf"

# Compile with bibliography
with-bib: $(DOC).tex
	@echo "Compiling LaTeX document with bibliography..."
	$(LATEX) -output-directory=$(BUILD_DIR) $(DOC).tex
	$(BIBTEX) $(BUILD_DIR)/$(DOC)
	$(LATEX) -output-directory=$(BUILD_DIR) $(DOC).tex
	$(LATEX) -output-directory=$(BUILD_DIR) $(DOC).tex
	@mv $(BUILD_DIR)/$(DOC).pdf .
	@echo "PDF generated: $(DOC).pdf"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(DOC).pdf
	@rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg
	@echo "Cleanup complete."

# View the PDF (requires xdg-open or evince)
view: $(DOC).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(DOC).pdf; \
	elif command -v evince > /dev/null; then \
		evince $(DOC).pdf; \
	else \
		echo "No PDF viewer found. Please open $(DOC).pdf manually."; \
	fi

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Compile the PDF (default)"
	@echo "  with-bib  - Compile with bibliography"
	@echo "  clean     - Remove auxiliary files and PDF"
	@echo "  view      - Open the PDF"
	@echo "  help      - Show this help message"

# Ensure build directory exists before compilation
$(DOC).pdf: | $(BUILD_DIR)

