# Thermal Image Circle Detection

A comprehensive Python application for detecting and analyzing circular patterns in thermal images using computer vision techniques.

## Features

- **Robust Circle Detection**: Advanced Hough Circle Transform with intelligent filtering
- **Multi-method ROI Detection**: Combines HSV color-based, intensity-based, and percentile-based approaches
- **Quality Assessment**: Evaluates circle completeness and edge quality
- **Batch Processing**: Process multiple images automatically
- **Comprehensive Reporting**: 
  - JSON format for machine-readable results
  - Human-readable text reports
  - Combined visualization of all detections
  - **Separate statistical charts** (no overlapping!)
    - Circle count bar chart
    - Radius distribution histogram
    - Radius box plot by image
    - Summary statistics table
- **Configurable Parameters**: Centralized `config.py` for easy parameter tuning

## Project Structure

```
Assignment 1/
├── main.py                    # Main entry point for batch processing
├── circle_detector.py         # Core circle detection logic
├── visualization.py           # Visualization and reporting functions
├── config.py                  # Centralized configuration parameters
├── batch_processor.py         # Original monolithic implementation (deprecated)
├── requirements.txt           # Python dependencies
├── Images/                    # Input thermal images
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Results/                   # Output directory (auto-generated)
    ├── visualizations/
    │   ├── all_results.png          # Combined detection results
    │   ├── circle_counts.png        # Bar chart of circle counts
    │   ├── radius_distribution.png  # Histogram of radii
    │   ├── radius_boxplot.png       # Box plot by image
    │   └── summary_table.png        # Summary statistics table
    ├── detection_report.json        # Machine-readable results
    ├── detection_report.txt         # Human-readable report
    ├── images/                      # Individual detection results
    ├── masks/                       # ROI mask visualizations
    ├── intermediate/                # Raw circle detections (before filtering)
    └── rejected_circles/            # Circles filtered out with reasons
```

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - opencv-python
   - numpy
   - matplotlib

## Usage

### Process All Images (Recommended)

Run the main script to process all images in the `Images/` directory:

```bash
python main.py
```

This will:
1. Process all images in the `Images/` directory
2. Detect complete circles in each image
3. Save individual detection results
4. Generate comprehensive reports and visualizations

### Process a Single Image

To process a single image with visualization:

```python
from main import process_image

detector, circles = process_image('Images/image1.png', save=True, display=True)
```

### Custom Parameters

Adjust detection parameters for your specific needs:

```python
from circle_detector import ImprovedCircleDetector

detector = ImprovedCircleDetector('Images/image1.png')
circles = detector.detect_complete_circles(
    min_radius=50,           # Minimum circle radius in pixels
    max_radius=200,          # Maximum circle radius in pixels
    quality_threshold=0.1    # Minimum quality score (0-1)
)
```

**Note**: Default parameters are defined in `config.py`. You can modify them there for global changes or pass them as arguments for per-call customization.

## Detection Algorithm

The circle detection process consists of several stages:

### 1. Preprocessing
- Gaussian blur to reduce noise
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement

### 2. ROI (Region of Interest) Detection
Combines three complementary approaches:
- **HSV color-based**: Detects warm colors (red/orange) typical in thermal images
- **Intensity-based**: Uses Otsu's and adaptive thresholding
- **Percentile-based**: Identifies the warmest 30% of pixels

### 3. Circle Detection
- Hough Circle Transform with strict parameters
- Detects potential circular patterns

### 4. Filtering & Validation
Circles are filtered based on:
- **Boundary check**: Must be completely within image bounds
- **ROI coverage**: At least 95% must be within detected ROI
- **Edge quality**: Evaluates edge strength along circle perimeter

### 5. Quality Scoring
Each circle receives a quality score based on:
- Edge continuity around the perimeter
- Sampling 36 points around the circle
- Checking for edge presence at each point

## Output Files

### Visualizations
All visualizations are saved as separate files to avoid overlapping:
- **all_results.png**: Grid view of all detection results
- **circle_counts.png**: Bar chart showing circle count per image
- **radius_distribution.png**: Histogram of radius distribution across all circles
- **radius_boxplot.png**: Box plot showing radius distribution by image
- **summary_table.png**: Summary statistics table

### Reports
- **detection_report.json**: Structured data for programmatic access
- **detection_report.txt**: Human-readable detailed report

### Debug Information
- **masks/**: ROI detection masks showing intermediate steps
- **intermediate/**: Raw circle detections before filtering
- **rejected_circles/**: Visual explanation of why circles were filtered out

## Parameters Reference

All default parameters are defined in `config.py` for easy centralized configuration.

### Circle Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_radius` | 50 | Minimum circle radius in pixels |
| `max_radius` | 200 | Maximum circle radius in pixels |
| `quality_threshold` | 0.1 | Minimum quality score (0-1) |
| `minDist` | 90 | Minimum distance between circle centers |
| `param1` | 100 | Canny edge detection threshold |
| `param2` | 40 | Accumulator threshold for circle detection |

### Filtering Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `margin` | -10 | Margin from image edges (negative allows near-edge) |
| `roi_coverage` | 0.95 | Minimum ROI coverage ratio (95%) |
| `edge_samples` | 36 | Number of points sampled around perimeter |

**To modify parameters globally**, edit `config.py`. Parameters can also be overridden per-call by passing them as arguments.

## Troubleshooting

### No circles detected
- Try lowering `quality_threshold` in `config.py` (e.g., 0.05)
- Adjust `min_radius` and `max_radius` based on expected circle sizes
- Check ROI masks in `Results/masks/` to verify ROI detection

### Too many false positives
- Increase `quality_threshold` in `config.py` (e.g., 0.3)
- Increase `param2` for stricter Hough detection
- Increase `minDist` to prevent overlapping detections

### Circles near edges rejected
- Adjust `margin` parameter in `config.py` (DETECTION_PARAMS)
- Lower `roi_coverage` threshold (but keep above 0.8)

### Modify parameters
Edit `config.py` to change default parameters globally, or pass them as arguments for per-call customization.

## Development

### Running Tests
```bash
# Process single image for testing
python -c "from main import process_image; process_image('Images/image1.png', display=True)"
```

### Extending Functionality
The modular structure makes it easy to extend:
- **circle_detector.py**: Modify detection algorithms
- **visualization.py**: Add new visualizations or export formats
- **main.py**: Customize batch processing workflow

## Technical Details

### Dependencies
- **OpenCV**: Image processing and circle detection
- **NumPy**: Numerical operations and array handling
- **Matplotlib**: Visualization and plotting

### Performance
- Processing time: ~2-5 seconds per image (depending on size and complexity)
- Memory usage: ~50-100 MB per image

## License

This project is provided for educational purposes.

## Author

Created for Advanced Vision Processing course (AVPR) - FIB

## Acknowledgments

- Uses OpenCV's Hough Circle Transform
- Inspired by thermal imaging analysis techniques
- CLAHE enhancement for improved contrast in thermal images

