#!/bin/bash
# Script to compile the LaTeX report

set -e  # Exit on error

echo "================================"
echo "LaTeX Report Compilation Script"
echo "================================"
echo ""

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found!"
    echo "Please install a LaTeX distribution:"
    echo "  - Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  - macOS: brew install --cask mactex"
    exit 1
fi

# Create build directory
mkdir -p build

echo "Step 1: First LaTeX pass..."
pdflatex -interaction=nonstopmode -output-directory=build report.tex > /dev/null 2>&1

echo "Step 2: Second LaTeX pass (for references)..."
pdflatex -interaction=nonstopmode -output-directory=build report.tex > /dev/null 2>&1

# Move PDF to main directory
if [ -f "build/report.pdf" ]; then
    mv build/report.pdf .
    echo ""
    echo "✓ Success! PDF generated: report.pdf"
    echo ""

    # Display file info
    if command -v du &> /dev/null; then
        SIZE=$(du -h report.pdf | cut -f1)
        echo "File size: $SIZE"
    fi

    if command -v pdfinfo &> /dev/null; then
        PAGES=$(pdfinfo report.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
        if [ ! -z "$PAGES" ]; then
            echo "Pages: $PAGES"
        fi
    fi

    echo ""
    echo "To view the report:"
    echo "  - Linux: xdg-open report.pdf"
    echo "  - macOS: open report.pdf"
    echo "  - Or use: make view"
else
    echo ""
    echo "✗ Error: PDF generation failed!"
    echo ""
    echo "Check build/report.log for details"
    exit 1
fi

