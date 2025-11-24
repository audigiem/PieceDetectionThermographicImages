# Circle Detection in Thermal Images - Assignment 1

This project implements multiple computer vision techniques to detect and delineate circular objects in thermal imagery.

## 📋 Overview

The assignment focuses on identifying complete circular objects in thermal images using various image processing and computer vision techniques including:
- Hough Circle Transform
- Contour-based detection
- Edge detection with circle fitting
- Blob detection
- Watershed segmentation
- Ensemble methods combining multiple techniques

## 🚀 Quick Start

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Code

#### Option 1: Batch Process All Images (Recommended)
Process all images in the `Images` folder automatically:
```bash
python batch_processor.py
```

This will:
- Process all images in the Images directory
- Save detected circles with annotations
- Generate comprehensive reports (JSON and text)
- Create comparison visualizations
- Generate statistical analysis plots

#### Option 2: Single Image Processing
Process individual images with the basic detector:
```bash
python circle_detection.py
```

Or use the advanced detector:
```bash
python advanced_circle_detection.py
```

## 📁 Project Structure

```
Assignment 1/
├── circle_detection.py          # Basic circle detection (Hough + Contours)
├── advanced_circle_detection.py # Advanced methods (Canny, Blob, Watershed, Ensemble)
├── batch_processor.py           # Batch processing script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── Images/                      # Input thermal images
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Results/                     # Generated after processing
    ├── images/                  # Detected images
    ├── visualizations/          # Comparison plots
    ├── detection_report.json    # Detailed JSON report
    └── detection_report.txt     # Human-readable report
```

## 🔧 Core Components

### 1. circle_detection.py
Basic circle detection implementation with two main methods:

**CircleDetector Class:**
- `detect_circles_hough()`: Uses Hough Circle Transform
- `detect_circles_contours()`: Contour-based detection with circularity filtering
- `preprocess_image()`: Image enhancement (Gaussian blur, CLAHE)
- `draw_circles()`: Visualize detected circles
- `get_circle_statistics()`: Calculate detection statistics

**Usage Example:**
```python
from circle_detection import CircleDetector

detector = CircleDetector("Images/image1.png")
circles = detector.detect_circles_hough()
detector.draw_circles(circles)
detector.save_result("output.png")
detector.display_results(circles)
```

### 2. advanced_circle_detection.py
Advanced detection techniques for challenging cases:

**AdvancedCircleDetector Class:**
- `detect_with_canny_and_fitting()`: Edge detection + ellipse fitting
- `detect_with_blob_detector()`: SimpleBlobDetector with circularity filters
- `detect_with_watershed()`: Watershed segmentation
- `ensemble_detection()`: Combines multiple methods for robust detection
- `enhance_thermal_image()`: Enhanced preprocessing for thermal images

**Usage Example:**
```python
from advanced_circle_detection import AdvancedCircleDetector

detector = AdvancedCircleDetector("Images/image1.png")
circles = detector.ensemble_detection(methods=['hough', 'canny', 'blob'])
detector.draw_circles(circles)
```

### 3. batch_processor.py
Comprehensive batch processing with reporting:

**BatchProcessor Class:**
- `process_all_images()`: Process entire image directory
- `process_single_image()`: Process with automatic method selection
- `generate_report()`: Create JSON and text reports
- `create_comparison_visualization()`: Generate comparison plots
- Statistical analysis and visualization

## 🎯 Detection Methods

### Hough Circle Transform
- **Pros**: Fast, robust to noise, works well for perfect circles
- **Cons**: Requires parameter tuning, may miss incomplete circles
- **Best for**: Clean thermal images with distinct circular objects

### Contour-based Detection
- **Pros**: Detects irregular circles, flexible shape matching
- **Cons**: Sensitive to noise and thresholding
- **Best for**: Images with clear boundaries

### Edge Detection + Fitting
- **Pros**: Good for partially occluded circles
- **Cons**: Computationally intensive
- **Best for**: Complex scenes with overlapping objects

### Blob Detection
- **Pros**: Excellent circularity filtering
- **Cons**: Requires good contrast
- **Best for**: High-contrast thermal spots

### Ensemble Method
- **Pros**: Most robust, combines strengths of all methods
- **Cons**: Slower processing time
- **Best for**: Production use, varied image conditions

## 📊 Output

### Detection Results
- **Annotated Images**: Circles marked with green outlines and red centers
- **Numbered Labels**: Each circle labeled for reference
- **Statistics**: Count, mean radius, radius range

### Reports
- **JSON Report**: Machine-readable detailed statistics
- **Text Report**: Human-readable summary
- **Visualizations**: Comparison grids and statistical plots

### Example Statistics:
```
File: image1.png
  Method: hough
  Circles detected: 5
  Mean radius: 32.4 px
  Std radius: 5.8 px
  Radius range: [25, 42] px
```

## 🔬 Technical Details

### Image Preprocessing
1. **Gaussian Blur**: Noise reduction (kernel size: 9x9)
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Bilateral Filter**: Edge-preserving smoothing (for advanced methods)
4. **Morphological Operations**: Closing and opening to clean binary masks

### Parameter Optimization
Key parameters for Hough Transform:
- `dp=1`: Inverse ratio of accumulator resolution
- `minDist=30`: Minimum distance between circle centers
- `param1=50`: Upper Canny threshold
- `param2=25`: Accumulator threshold
- `minRadius=5`, `maxRadius=200`: Size constraints

### Circle Validation
Circularity metric: `C = 4π × Area / Perimeter²`
- Perfect circle: C = 1.0
- Acceptance threshold: C > 0.7

## 💡 Tips for Best Results

1. **For noisy images**: Use ensemble detection
2. **For low contrast**: Increase CLAHE clip limit
3. **For small circles**: Decrease `minRadius` parameter
4. **For large circles**: Increase `maxRadius` parameter
5. **For overlapping circles**: Adjust `minDist` parameter

## 🛠️ Customization

### Adjust Detection Parameters
Edit the parameters in `batch_processor.py`:
```python
circles = detector.detect_circles_hough(
    min_dist=30,      # Increase to avoid close detections
    param1=50,        # Edge detection sensitivity
    param2=25,        # Circle detection threshold (lower = more circles)
    min_radius=5,     # Minimum circle size
    max_radius=200    # Maximum circle size
)
```

### Choose Detection Method
In `batch_processor.py`, change the method parameter:
```python
processor.process_all_images(method='auto')  # Options: 'auto', 'hough', 'advanced'
```

## 📈 Performance

Typical processing times (per image):
- **Hough Transform**: 0.1-0.3 seconds
- **Contour Detection**: 0.2-0.4 seconds
- **Ensemble Method**: 0.5-1.0 seconds

## 🐛 Troubleshooting

**No circles detected:**
- Try lowering `param2` parameter
- Check image contrast
- Verify image is not corrupted

**Too many false positives:**
- Increase `param2` parameter
- Increase `minDist` parameter
- Adjust circularity threshold

**Slow processing:**
- Use 'hough' method instead of 'auto'
- Reduce `maxRadius` if circles are small

## 📝 Assignment Deliverables

This implementation provides:
1. ✅ Circle detection and delineation
2. ✅ Multiple detection methods
3. ✅ Batch processing capability
4. ✅ Comprehensive reporting
5. ✅ Visual comparison
6. ✅ Statistical analysis
7. ✅ Well-documented code

## 🎓 Learning Objectives Covered

- Image preprocessing techniques
- Circle detection algorithms
- Contour analysis
- Feature extraction
- Performance evaluation
- Visualization techniques

## 📚 References

- OpenCV Documentation: https://docs.opencv.org/
- Hough Circle Transform: https://en.wikipedia.org/wiki/Circle_Hough_Transform
- Computer Vision: Algorithms and Applications (Szeliski)

## 👨‍💻 Author

Assignment 1 - AVPR Course
FIB - Computer Vision and Pattern Recognition

---

**Note**: Ensure thermal images are placed in the `Images/` directory before running the scripts.

