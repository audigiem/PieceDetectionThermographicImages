"""
Improved Circle Detection for Thermal Images
This version includes better filtering to detect only COMPLETE circles
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ImprovedCircleDetector:
    """Improved circle detector with strict filtering for complete circles only."""

    def __init__(self, image_path):
        """Initialize with image path."""
        self.image_path = image_path
        self.original = cv2.imread(str(image_path))
        if self.original is None:
            raise ValueError(f"Cannot load image: {image_path}")

        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.result = self.original.copy()
        self.height, self.width = self.gray.shape

    def preprocess_thermal_image(self):
        """Enhanced preprocessing for thermal images."""
        # Strong Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (15, 15), 3)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced

    def is_circle_complete(self, x, y, radius, margin=10):
        """
        Check if a circle is completely within the image bounds.

        Args:
            x, y: Circle center
            radius: Circle radius
            margin: Margin from image edges

        Returns:
            True if circle is complete, False otherwise
        """
        x = int(x)
        y = int(y)
        if (x - radius < margin or
            x + radius > self.width - margin or
            y - radius < margin or
            y + radius > self.height - margin):
            return False
        return True

    def calculate_circle_quality(self, x, y, radius):
        """
        Calculate quality score for a detected circle based on edge strength.

        Args:
            x, y: Circle center
            radius: Circle radius

        Returns:
            Quality score (0-1)
        """
        # Get edges
        enhanced = self.preprocess_thermal_image()
        edges = cv2.Canny(enhanced, 50, 150)

        # Sample points along the circle perimeter
        num_points = 36  # Sample 36 points around the circle
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

        edge_votes = 0
        for angle in angles:
            # Calculate point on circle perimeter
            px = int(x + radius * np.cos(angle))
            py = int(y + radius * np.sin(angle))

            # Check if point is within bounds
            if 0 <= px < self.width and 0 <= py < self.height:
                # Check if there's an edge at this point (with small tolerance)
                region = edges[max(0, py-2):min(self.height, py+3),
                             max(0, px-2):min(self.width, px+3)]
                if np.any(region > 0):
                    edge_votes += 1

        # Calculate quality as ratio of edge points found
        quality = edge_votes / num_points
        return quality

    def remove_overlapping_circles(self, circles, overlap_threshold=0.3):
        """
        Remove overlapping circles, keeping the ones with better quality.

        Args:
            circles: List of (x, y, radius, quality) tuples
            overlap_threshold: Maximum allowed overlap ratio

        Returns:
            Filtered list of circles
        """
        if len(circles) == 0:
            return []

        # Sort by quality (descending)
        circles = sorted(circles, key=lambda c: c[3], reverse=True)

        keep = []
        for circle in circles:
            x1, y1, r1, q1 = circle

            # Check against all kept circles
            overlap = False
            for kept_circle in keep:
                x2, y2, r2, q2 = kept_circle

                # Calculate distance between centers
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                # Check for overlap
                if dist < (r1 + r2) * (1 - overlap_threshold):
                    overlap = True
                    break

            if not overlap:
                keep.append(circle)

        return keep

    def detect_complete_circles(self, min_radius=50, max_radius=200,
                                quality_threshold=0.4):
        """
        Detect only complete circles with high quality.

        Args:
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            quality_threshold: Minimum quality score (0-1)

        Returns:
            List of detected complete circles
        """
        enhanced = self.preprocess_thermal_image()

        # Use more strict Hough parameters
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=80,  # Increased minimum distance between circles
            param1=100,   # Higher Canny threshold
            param2=40,    # Higher accumulator threshold
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is None:
            return []

        circles = np.uint16(np.around(circles[0]))

        # Filter circles
        valid_circles = []

        for circle in circles:
            x, y, radius = circle

            # Check if circle is complete (within image bounds)
            if not self.is_circle_complete(x, y, radius):
                continue

            # Calculate quality score
            quality = self.calculate_circle_quality(x, y, radius)

            # Keep only high-quality circles
            if quality >= quality_threshold:
                valid_circles.append((int(x), int(y), int(radius), quality))

        # Remove overlapping circles
        valid_circles = self.remove_overlapping_circles(valid_circles, overlap_threshold=0.3)

        # Return without quality scores
        return [(x, y, r) for x, y, r, q in valid_circles]

    def draw_circles(self, circles, color=(0, 255, 0), thickness=3):
        """Draw detected circles on result image."""
        for idx, (x, y, radius) in enumerate(circles, 1):
            # Draw circle outline
            cv2.circle(self.result, (x, y), radius, color, thickness)
            # Draw center point
            cv2.circle(self.result, (x, y), 5, (0, 0, 255), -1)
            # Add number annotation
            cv2.putText(self.result, str(idx), (x - 15, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    def visualize_results(self, circles):
        """Display original and result side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original
        axes[0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Thermal Image')
        axes[0].axis('off')

        # Result
        axes[1].imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Detected Complete Circles: {len(circles)}')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


def process_image(image_path, save=True, display=True):
    """
    Process a single image to detect complete circles.

    Args:
        image_path: Path to thermal image
        save: Whether to save result
        display: Whether to display result
    """
    print(f"\nProcessing: {Path(image_path).name}")

    detector = ImprovedCircleDetector(image_path)

    # Detect complete circles
    circles = detector.detect_complete_circles(
        min_radius=50,      # Adjust based on expected circle size
        max_radius=200,     # Adjust based on expected circle size
        quality_threshold=0.1  # Minimum quality (0.4 = 40% of perimeter must have edges)
    )

    print(f"✓ Detected {len(circles)} complete circles")

    if len(circles) > 0:
        detector.draw_circles(circles)

        # Print details
        radii = [r for _, _, r in circles]
        print(f"  Mean radius: {np.mean(radii):.1f} px")
        print(f"  Radius range: [{min(radii)}, {max(radii)}] px")

    # Save result
    if save:
        output_dir = Path('Results/images')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_path), detector.result)
        print(f"  Saved to: {output_path}")

    # Display
    if display:
        detector.visualize_results(circles)

    return detector, circles


def process_all_images(images_dir='Images', save=True):
    """Process all images in directory."""
    images_path = Path(images_dir)
    image_files = sorted(images_path.glob('*.png'))
    image_files.extend(sorted(images_path.glob('*.jpg')))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print("="*70)
    print("IMPROVED CIRCLE DETECTION - COMPLETE CIRCLES ONLY")
    print("="*70)
    print(f"\nProcessing {len(image_files)} images...\n")

    results = []

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}]", end=" ")
        try:
            detector, circles = process_image(img_path, save=save, display=False)
            results.append({
                'filename': img_path.name,
                'count': len(circles),
                'circles': circles
            })
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for result in results:
        print(f"{result['filename']:20} -> {result['count']} complete circles")

    print(f"\nTotal circles detected: {sum(r['count'] for r in results)}")


# Example usage
if __name__ == "__main__":
    # Process single image with visualization
    # process_image('Images/image1.png', save=True, display=True)

    # Process all images
    process_all_images('Images', save=True)