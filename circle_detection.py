"""
Circle Detection in Thermal Images - Assignment 1
This script identifies and delineates individual complete circular objects in thermal imagery.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class CircleDetector:
    """Class for detecting circles in thermal images."""
    
    def __init__(self, image_path):
        """
        Initialize the CircleDetector.
        
        Args:
            image_path: Path to the thermal image
        """
        self.image_path = image_path
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.result_image = self.original_image.copy()
        
    def preprocess_image(self):
        """
        Preprocess the thermal image for better circle detection.
        
        Returns:
            Preprocessed grayscale image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_circles_hough(self, min_dist=50, param1=50, param2=30, 
                            min_radius=10, max_radius=100):
        """
        Detect circles using Hough Circle Transform.
        
        Args:
            min_dist: Minimum distance between detected circle centers
            param1: Upper threshold for Canny edge detection
            param2: Accumulator threshold for circle centers
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            
        Returns:
            Detected circles as numpy array
        """
        preprocessed = self.preprocess_image()
        
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        return circles
    
    def detect_circles_contours(self, threshold_method='adaptive'):
        """
        Detect circles using contour detection method.
        
        Args:
            threshold_method: 'adaptive', 'otsu', or 'binary'
            
        Returns:
            List of detected circles as (x, y, radius) tuples
        """
        preprocessed = self.preprocess_image()
        
        # Apply thresholding
        if threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                preprocessed, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        elif threshold_method == 'otsu':
            _, binary = cv2.threshold(
                preprocessed, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                preprocessed, 127, 255,
                cv2.THRESH_BINARY_INV
            )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        circles = []
        for contour in contours:
            # Calculate circularity
            area = cv2.contourArea(contour)
            if area < 100:  # Filter out small contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # If circularity is close to 1, it's likely a circle
            if circularity > 0.7:
                # Get minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circles.append((int(x), int(y), int(radius)))
        
        return circles
    
    def draw_circles(self, circles, color=(0, 255, 0), thickness=2):
        """
        Draw detected circles on the result image.
        
        Args:
            circles: Detected circles
            color: Color for drawing circles (BGR format)
            thickness: Line thickness
        """
        if circles is not None:
            if isinstance(circles, np.ndarray) and len(circles.shape) == 3:
                # Hough circles format
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    # Draw circle outline
                    cv2.circle(self.result_image, center, radius, color, thickness)
                    # Draw center point
                    cv2.circle(self.result_image, center, 3, (0, 0, 255), -1)
            else:
                # Contour circles format
                for (x, y, radius) in circles:
                    center = (x, y)
                    # Draw circle outline
                    cv2.circle(self.result_image, center, radius, color, thickness)
                    # Draw center point
                    cv2.circle(self.result_image, center, 3, (0, 0, 255), -1)
    
    def annotate_circles(self, circles):
        """
        Annotate each detected circle with a number.
        
        Args:
            circles: Detected circles
        """
        if circles is not None:
            if isinstance(circles, np.ndarray) and len(circles.shape) == 3:
                circles_list = circles[0, :]
            else:
                circles_list = circles
                
            for idx, circle in enumerate(circles_list, 1):
                if isinstance(circle, np.ndarray):
                    x, y = circle[0], circle[1]
                else:
                    x, y = circle[0], circle[1]
                    
                # Put text annotation
                cv2.putText(
                    self.result_image, 
                    str(idx), 
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 0), 
                    2
                )
    
    def get_circle_statistics(self, circles):
        """
        Calculate statistics for detected circles.
        
        Args:
            circles: Detected circles
            
        Returns:
            Dictionary with statistics
        """
        if circles is None or len(circles) == 0:
            return {"count": 0}
        
        if isinstance(circles, np.ndarray) and len(circles.shape) == 3:
            radii = circles[0, :, 2]
        else:
            radii = np.array([c[2] for c in circles])
        
        stats = {
            "count": len(radii),
            "mean_radius": np.mean(radii),
            "std_radius": np.std(radii),
            "min_radius": np.min(radii),
            "max_radius": np.max(radii)
        }
        
        return stats
    
    def save_result(self, output_path):
        """
        Save the result image with detected circles.
        
        Args:
            output_path: Path to save the result image
        """
        cv2.imwrite(str(output_path), self.result_image)
        print(f"Result saved to {output_path}")
    
    def display_results(self, circles=None):
        """
        Display original image and result side by side.
        
        Args:
            circles: Detected circles (for showing statistics)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Thermal Image')
        axes[0].axis('off')
        
        # Result image
        axes[1].imshow(cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Detected Circles')
        axes[1].axis('off')
        
        # Add statistics as text
        if circles is not None:
            stats = self.get_circle_statistics(circles)
            stats_text = f"Circles detected: {stats['count']}"
            if stats['count'] > 0:
                stats_text += f"\nMean radius: {stats['mean_radius']:.1f} px"
                stats_text += f"\nRadius range: [{stats['min_radius']}, {stats['max_radius']}] px"
            
            plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def process_single_image(image_path, method='hough', display=True, save=True):
    """
    Process a single thermal image to detect circles.
    
    Args:
        image_path: Path to the image
        method: Detection method ('hough' or 'contours')
        display: Whether to display results
        save: Whether to save results
        
    Returns:
        CircleDetector instance with results
    """
    print(f"\nProcessing: {image_path}")
    detector = CircleDetector(image_path)
    
    if method == 'hough':
        circles = detector.detect_circles_hough(
            min_dist=30,
            param1=50,
            param2=25,
            min_radius=5,
            max_radius=200
        )
    else:
        circles = detector.detect_circles_contours(threshold_method='adaptive')
    
    if circles is not None and len(circles) > 0:
        detector.draw_circles(circles)
        detector.annotate_circles(circles)
        stats = detector.get_circle_statistics(circles)
        print(f"✓ Found {stats['count']} circles")
        if stats['count'] > 0:
            print(f"  Mean radius: {stats['mean_radius']:.1f} px")
    else:
        print("✗ No circles detected")
    
    if save:
        output_dir = Path(image_path).parent.parent / 'Results'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"result_{Path(image_path).name}"
        detector.save_result(output_path)
    
    if display:
        detector.display_results(circles)
    
    return detector


def process_all_images(images_dir, method='hough', display=False, save=True):
    """
    Process all images in a directory.
    
    Args:
        images_dir: Directory containing images
        method: Detection method ('hough' or 'contours')
        display: Whether to display results for each image
        save: Whether to save results
    """
    images_path = Path(images_dir)
    image_files = sorted(images_path.glob('*.png'))
    
    if not image_files:
        print(f"No PNG images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    results = []
    
    for img_file in image_files:
        try:
            detector = process_single_image(img_file, method=method, 
                                          display=display, save=save)
            results.append((img_file.name, detector))
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for img_name, detector in results:
        circles = detector.detect_circles_hough() if method == 'hough' else detector.detect_circles_contours()
        stats = detector.get_circle_statistics(circles)
        print(f"{img_name:20} -> {stats['count']} circles detected")


def compare_methods(image_path):
    """
    Compare Hough and Contours detection methods on the same image.
    
    Args:
        image_path: Path to the image
    """
    print(f"\nComparing detection methods on: {image_path}\n")
    
    # Hough method
    print("Method 1: Hough Circle Transform")
    detector1 = CircleDetector(image_path)
    circles1 = detector1.detect_circles_hough()
    detector1.draw_circles(circles1, color=(0, 255, 0))
    stats1 = detector1.get_circle_statistics(circles1)
    
    # Contours method
    print("\nMethod 2: Contour-based Detection")
    detector2 = CircleDetector(image_path)
    circles2 = detector2.detect_circles_contours()
    detector2.draw_circles(circles2, color=(255, 0, 0))
    stats2 = detector2.get_circle_statistics(circles2)
    
    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    axes[0].imshow(cv2.cvtColor(detector1.original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(detector1.result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Hough Transform\n({stats1["count"]} circles)')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(detector2.result_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Contour-based\n({stats2["count"]} circles)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration
    IMAGES_DIR = "Images"
    
    # Example 1: Process all images using Hough method
    print("="*60)
    print("CIRCLE DETECTION IN THERMAL IMAGES")
    print("="*60)
    process_all_images(IMAGES_DIR, method='hough', display=False, save=True)
    
    # Example 2: Process a single image with display
    # Uncomment to process specific image with visualization
    # process_single_image("Images/image1.png", method='hough', display=True)
    
    # Example 3: Compare methods on a single image
    # Uncomment to compare detection methods
    # compare_methods("Images/image1.png")

