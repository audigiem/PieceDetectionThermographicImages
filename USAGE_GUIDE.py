"""
USAGE GUIDE - Circle Detection in Thermal Images
=================================================

This guide provides complete instructions for using the circle detection system.
"""

# =============================================================================
# INSTALLATION
# =============================================================================

"""
1. Ensure Python 3.7+ is installed
2. Navigate to the project directory
3. Install dependencies:

   pip install -r requirements.txt

This installs:
- opencv-python (image processing)
- numpy (numerical operations)
- matplotlib (visualization)
- scipy (scientific computing)
- PyPDF2 (PDF reading)
"""

# =============================================================================
# QUICK START
# =============================================================================

"""
OPTION 1: BATCH PROCESSING (RECOMMENDED)
-----------------------------------------
Process all images in the Images folder at once:

   python batch_processor.py

This will:
✓ Process all images automatically
✓ Save results to Results/ directory
✓ Generate comprehensive reports
✓ Create visualizations and statistics


OPTION 2: INTERACTIVE MENU
---------------------------
Run the interactive demo with a menu:

   python main_demo.py

Choose from multiple options:
1. Batch process all images
2. Process single image interactively
3. Compare detection methods
4. Interactive parameter tuning
5. Ensemble detection demo
6. Processing steps visualization
7. Quick test


OPTION 3: SINGLE IMAGE
----------------------
Process a specific image programmatically:

   python circle_detection.py

Or edit the __main__ section to specify your image.
"""

# =============================================================================
# CODE EXAMPLES
# =============================================================================

# Example 1: Basic Circle Detection
# ----------------------------------
from circle_detection import CircleDetector

# Load and detect circles
detector = CircleDetector("Images/image1.png")
circles = detector.detect_circles_hough(
    min_dist=30,      # Minimum distance between circles
    param1=50,        # Canny edge detection threshold
    param2=25,        # Accumulator threshold (lower = more circles)
    min_radius=5,     # Minimum circle radius
    max_radius=200    # Maximum circle radius
)

# Draw and save results
detector.draw_circles(circles)
detector.annotate_circles(circles)
detector.save_result("Results/output.png")

# Get statistics
stats = detector.get_circle_statistics(circles)
print(f"Detected {stats['count']} circles")
print(f"Mean radius: {stats['mean_radius']:.1f} px")

# Display results
detector.display_results(circles)


# Example 2: Alternative Contour-based Detection
# -----------------------------------------------
from circle_detection import CircleDetector

detector = CircleDetector("Images/image1.png")
circles = detector.detect_circles_contours(threshold_method='adaptive')
detector.draw_circles(circles)
detector.save_result("Results/contour_output.png")


# Example 3: Advanced Ensemble Detection
# ---------------------------------------
from advanced_circle_detection import AdvancedCircleDetector

detector = AdvancedCircleDetector("Images/image1.png")

# Use multiple detection methods combined
circles = detector.ensemble_detection(
    methods=['hough', 'canny', 'blob']
)

detector.draw_circles(circles)
print(f"Ensemble detected {len(circles)} circles")


# Example 4: Visualize Processing Steps
# --------------------------------------
from advanced_circle_detection import AdvancedCircleDetector

detector = AdvancedCircleDetector("Images/image1.png")
detector.visualize_processing_steps()


# Example 5: Compare Detection Methods
# -------------------------------------
from circle_detection import compare_methods

compare_methods("Images/image1.png")


# Example 6: Batch Processing with Custom Settings
# -------------------------------------------------
from batch_processor import BatchProcessor

processor = BatchProcessor("Images", "CustomResults")
processor.process_all_images(method='auto')  # or 'hough' or 'advanced'


# Example 7: Process Single Image from Batch Processor
# -----------------------------------------------------
from batch_processor import BatchProcessor

processor = BatchProcessor("Images", "Results")
result = processor.process_single_image("Images/image1.png", method='hough')

print(f"Circles found: {result['circle_count']}")
if result['circle_count'] > 0:
    print(f"Mean radius: {result['mean_radius']:.2f} px")
    print(f"Circle positions: {result['circles']}")


# Example 8: Parameter Tuning
# ----------------------------
from advanced_circle_detection import interactive_parameter_tuning

# Opens interactive window showing different parameter combinations
interactive_parameter_tuning("Images/image1.png")


# =============================================================================
# PARAMETER TUNING GUIDE
# =============================================================================

"""
Hough Circle Transform Parameters:
-----------------------------------

1. dp (resolution ratio)
   - Default: 1
   - Lower = higher resolution but slower
   - Higher = faster but less accurate

2. minDist (minimum distance between circles)
   - Too low: Multiple detections of same circle
   - Too high: Missing nearby circles
   - Suggested: 20-50 pixels

3. param1 (Canny edge threshold)
   - Default: 50
   - Higher = fewer edges detected
   - Lower = more edges, may increase false positives
   - Suggested range: 30-100

4. param2 (accumulator threshold)
   - Default: 25-30
   - Lower = more circles (including false positives)
   - Higher = fewer circles (may miss some)
   - Most important parameter for tuning
   - Suggested range: 15-40

5. minRadius / maxRadius
   - Set based on expected circle sizes in your images
   - Use None for no limit (slower)
   - Narrower range = faster processing


Tips for Different Scenarios:
------------------------------

For NOISY images:
- Increase param1 (e.g., 60-80)
- Increase param2 (e.g., 30-35)
- Use Gaussian blur with larger kernel

For LOW CONTRAST images:
- Decrease param2 (e.g., 15-20)
- Use CLAHE enhancement (already included)
- Try advanced ensemble method

For SMALL circles:
- Decrease minRadius
- Decrease minDist
- Increase image resolution if possible

For OVERLAPPING circles:
- Increase minDist
- Use contour-based detection instead
- Try watershed segmentation

For PARTIAL/INCOMPLETE circles:
- Use Canny + circle fitting method
- Lower circularity threshold
- Try contour-based detection
"""


# =============================================================================
# OUTPUT STRUCTURE
# =============================================================================

"""
After running batch_processor.py, the Results/ directory contains:

Results/
├── images/
│   ├── detected_image1.png      # Images with circles drawn
│   ├── detected_image2.png
│   └── ...
│
├── visualizations/
│   ├── all_results.png          # Grid comparison of all images
│   └── statistics.png           # Statistical plots
│
├── detection_report.json        # Detailed JSON report
└── detection_report.txt         # Human-readable summary


JSON Report Format:
-------------------
{
  "timestamp": "2025-11-24T...",
  "total_images": 7,
  "successful": 7,
  "failed": 0,
  "total_circles_detected": 25,
  "results": [
    {
      "filename": "image1.png",
      "method": "hough",
      "status": "success",
      "circle_count": 5,
      "mean_radius": 32.4,
      "std_radius": 5.8,
      "min_radius": 25,
      "max_radius": 42
    },
    ...
  ]
}
"""


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
Problem: No circles detected
Solutions:
1. Lower param2 (e.g., from 30 to 20)
2. Check if image loaded correctly:
   img = cv2.imread("path/to/image.png")
   print(img.shape)  # Should show (height, width, channels)
3. Try different detection method (contours or ensemble)
4. Visualize preprocessing steps to check image quality

Problem: Too many false positives
Solutions:
1. Increase param2 (e.g., from 25 to 35)
2. Increase minDist
3. Narrow minRadius/maxRadius range
4. Check circularity threshold in contour detection

Problem: Slow processing
Solutions:
1. Use 'hough' method instead of 'auto' or 'advanced'
2. Reduce maxRadius if circles are small
3. Process smaller image regions
4. Increase minDist to reduce candidates

Problem: Import errors
Solutions:
1. Ensure all dependencies installed:
   pip install -r requirements.txt
2. Check Python version (need 3.7+):
   python --version
3. Use virtual environment:
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate  # Windows
   pip install -r requirements.txt

Problem: Images not found
Solutions:
1. Check Images/ directory exists
2. Verify image format (PNG or JPG)
3. Use absolute paths if relative paths fail
4. Check file permissions
"""


# =============================================================================
# ADVANCED USAGE
# =============================================================================

"""
Custom Detection Pipeline:
--------------------------
"""

import cv2
import numpy as np
from circle_detection import CircleDetector

class CustomDetector(CircleDetector):
    """Custom detector with your own preprocessing."""

    def custom_preprocess(self):
        """Add your custom preprocessing here."""
        # Example: More aggressive noise reduction
        blurred = cv2.bilateralFilter(self.gray, 15, 80, 80)

        # Example: Custom thresholding
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        return binary

    def detect_with_custom_method(self):
        """Your custom detection logic."""
        preprocessed = self.custom_preprocess()

        # Use preprocessed image with Hough
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=60,
            param2=20,
            minRadius=10,
            maxRadius=100
        )

        return circles


# Usage:
detector = CustomDetector("Images/image1.png")
circles = detector.detect_with_custom_method()
detector.draw_circles(circles)


"""
Integrating with Your Own Pipeline:
------------------------------------
"""

def my_processing_pipeline(image_path, output_path):
    """Example of integrating circle detection in your pipeline."""

    # 1. Load image
    detector = CircleDetector(image_path)

    # 2. Detect circles
    circles = detector.detect_circles_hough()

    # 3. Filter circles by some criteria
    if circles is not None and len(circles.shape) == 3:
        filtered_circles = []
        for circle in circles[0]:
            x, y, r = circle
            # Example: Only keep circles in certain region
            if 100 < x < 500 and 100 < y < 500:
                # Example: Only keep certain sizes
                if 20 < r < 80:
                    filtered_circles.append(circle)

        # 4. Draw filtered circles
        detector.draw_circles(np.array([filtered_circles]))

    # 5. Save result
    detector.save_result(output_path)

    # 6. Return data for further processing
    return {
        'circles': circles,
        'image_shape': detector.original_image.shape,
        'output': output_path
    }


# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

"""
For Processing Many Images:
---------------------------
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from circle_detection import CircleDetector

def process_one_image(img_path):
    """Process single image - can be parallelized."""
    try:
        detector = CircleDetector(str(img_path))
        circles = detector.detect_circles_hough()

        # Save result
        output_path = Path("Results") / f"result_{img_path.name}"
        detector.draw_circles(circles)
        detector.save_result(output_path)

        stats = detector.get_circle_statistics(circles)
        return img_path.name, stats['count']
    except Exception as e:
        return img_path.name, f"Error: {e}"


def parallel_processing(images_dir, max_workers=4):
    """Process multiple images in parallel."""
    image_paths = list(Path(images_dir).glob("*.png"))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_one_image, image_paths)

    for filename, result in results:
        print(f"{filename}: {result}")


# Usage:
# parallel_processing("Images", max_workers=4)


# =============================================================================
# END OF USAGE GUIDE
# =============================================================================

print(__doc__)

