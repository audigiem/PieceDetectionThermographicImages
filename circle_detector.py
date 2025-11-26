"""
Circle Detector Module
Contains the main circle detection logic for thermal images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import DETECTION_PARAMS, PREPROCESSING_PARAMS, ROI_PARAMS, DIRECTORIES


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
        blurred = cv2.GaussianBlur(
            self.gray,
            PREPROCESSING_PARAMS["gaussian_kernel"],
            PREPROCESSING_PARAMS["gaussian_sigma"],
        )

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESSING_PARAMS["clahe_clip_limit"],
            tileGridSize=PREPROCESSING_PARAMS["clahe_tile_size"],
        )
        enhanced = clahe.apply(blurred)

        return enhanced

    def define_ROI_thermal(self, show=True):
        """Enhanced ROI detection for thermal images using multiple approaches."""
        # Method 1: HSV color-based detection (your original approach)
        hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)

        # Define color range for warm colors (red/orange)
        lower_warm = np.array(ROI_PARAMS["hsv_lower_1"])
        upper_warm = np.array(ROI_PARAMS["hsv_upper_1"])
        mask1 = cv2.inRange(hsv, lower_warm, upper_warm)

        lower_warm2 = np.array(ROI_PARAMS["hsv_lower_2"])
        upper_warm2 = np.array(ROI_PARAMS["hsv_upper_2"])
        mask2 = cv2.inRange(hsv, lower_warm2, upper_warm2)

        hsv_mask = cv2.bitwise_or(mask1, mask2)

        # Method 2: Intensity-based thresholding on grayscale
        # Detect the warmest regions based on intensity
        blur_gray = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # Use Otsu's thresholding to automatically find optimal threshold
        _, otsu_mask = cv2.threshold(
            blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Also try adaptive thresholding for local variations
        adaptive_mask = cv2.adaptiveThreshold(
            blur_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )

        # Method 3: Top percentile intensity thresholding
        # Find pixels in top N% of intensity values
        threshold_value = float(
            np.percentile(self.gray, ROI_PARAMS["intensity_percentile"])
        )
        _, percentile_mask = cv2.threshold(
            self.gray, threshold_value, 255, cv2.THRESH_BINARY
        )

        # Combine all methods
        # Give more weight to HSV-based detection for thermal images
        combined_mask = cv2.bitwise_or(hsv_mask, percentile_mask)
        combined_mask = cv2.bitwise_or(combined_mask, otsu_mask)

        # Clean up the mask with morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, ROI_PARAMS["morph_kernel_size"]
        )
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Fill small holes
        kernel_fill = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, ROI_PARAMS["fill_kernel_size"]
        )
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fill)

        # Optional: visualize all masks
        if show:
            mask_dir = Path(DIRECTORIES["masks_output"])
            if not mask_dir.exists():
                mask_dir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            axes[0, 0].imshow(hsv_mask, cmap="gray")
            axes[0, 0].set_title("HSV Color Mask")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(otsu_mask, cmap="gray")
            axes[0, 1].set_title("Otsu Threshold Mask")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(percentile_mask, cmap="gray")
            axes[0, 2].set_title("Top 30% Intensity Mask")
            axes[0, 2].axis("off")

            axes[1, 0].imshow(adaptive_mask, cmap="gray")
            axes[1, 0].set_title("Adaptive Threshold Mask")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(combined_mask, cmap="gray")
            axes[1, 1].set_title("Final Combined ROI Mask")
            axes[1, 1].axis("off")

            # Show original for reference
            axes[1, 2].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title("Original Image")
            axes[1, 2].axis("off")

            plt.tight_layout()
            plt.savefig(mask_dir / f"roi_masks_{Path(self.image_path).name}")
            plt.close()
            print(
                f"  Saved ROI mask visualizations to: {mask_dir}/roi_masks_{Path(self.image_path).name}"
            )

        return combined_mask

    def define_ROI(self, show=False):
        """Backward compatibility - calls the enhanced thermal ROI method."""
        return self.define_ROI_thermal(show)

    def is_circle_complete(self, x, y, radius, mask, margin=None):
        """
        Check if a circle is completely within the image bounds.

        Args:
            x, y: Circle center
            radius: Circle radius
            mask: ROI mask
            margin: Margin from image edges (uses config default if None)

        Returns:
            True if circle is complete, False otherwise
        """
        if margin is None:
            margin = DETECTION_PARAMS["margin"]

        x = int(x)
        y = int(y)
        radius = int(radius)
        if (
            x - radius < margin
            or x + radius > self.width - margin
            or y - radius < margin
            or y + radius > self.height - margin
        ):
            print(
                f"Circle at ({x}, {y}) with radius {radius} rejected: too close to image edge."
            )
            return False

        # Check if circle is within ROI
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (x, y), radius, 255, thickness=-1)
        intersection = cv2.bitwise_and(mask, circle_mask)
        circle_area = np.pi * (radius**2)
        intersection_area = cv2.countNonZero(intersection)
        roi_coverage = DETECTION_PARAMS["roi_coverage"]
        if intersection_area < circle_area * roi_coverage:
            print(
                f"Circle at ({x}, {y}) with radius {radius} rejected: insufficient ROI coverage."
            )
            # save rejected circle and intersection using matplotlib
            rejected_dir = Path(DIRECTORIES["rejected_output"])
            if not rejected_dir.exists():
                rejected_dir.mkdir(parents=True, exist_ok=True)
            temp_result = self.result.copy()
            cv2.circle(temp_result, (x, y), radius, (0, 0, 255), 3)
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")
            axes[0, 1].imshow(mask, cmap="gray")
            axes[0, 1].set_title("ROI Mask")
            axes[0, 1].axis("off")
            axes[1, 0].imshow(cv2.cvtColor(temp_result, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("Rejected Circle")
            axes[1, 0].axis("off")
            axes[1, 1].imshow(intersection, cmap="gray")
            axes[1, 1].set_title("Circle & ROI Intersection")
            axes[1, 1].axis("off")
            plt.tight_layout()
            plt.savefig(
                rejected_dir
                / f"rejected_circle_{x}_{y}_{radius}_{Path(self.image_path).name}"
            )
            plt.close()
            print(
                f"  Saved rejected circle details to: {rejected_dir}/rejected_circle_{x}_{y}_{radius}_{Path(self.image_path).name}"
            )
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
        num_points = DETECTION_PARAMS["edge_samples"]
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        edge_votes = 0
        for angle in angles:
            # Calculate point on circle perimeter
            px = int(x + radius * np.cos(angle))
            py = int(y + radius * np.sin(angle))

            # Check if point is within bounds
            if 0 <= px < self.width and 0 <= py < self.height:
                # Check if there's an edge at this point (with small tolerance)
                region = edges[
                    max(0, py - 2) : min(self.height, py + 3),
                    max(0, px - 2) : min(self.width, px + 3),
                ]
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
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                # Check for overlap
                if dist < (r1 + r2) * (1 - overlap_threshold):
                    overlap = True
                    break

            if not overlap:
                keep.append(circle)

        return keep

    def detect_complete_circles(
        self, min_radius=None, max_radius=None, quality_threshold=None
    ):
        """
        Detect only complete circles with high quality.

        Args:
            min_radius: Minimum circle radius (uses config default if None)
            max_radius: Maximum circle radius (uses config default if None)
            quality_threshold: Minimum quality score 0-1 (uses config default if None)

        Returns:
            List of detected complete circles
        """
        # Use config defaults if not specified
        if min_radius is None:
            min_radius = DETECTION_PARAMS["min_radius"]
        if max_radius is None:
            max_radius = DETECTION_PARAMS["max_radius"]
        if quality_threshold is None:
            quality_threshold = DETECTION_PARAMS["quality_threshold"]

        enhanced = self.preprocess_thermal_image()

        mask = self.define_ROI_thermal(show=True)

        # Use Hough parameters from config
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=DETECTION_PARAMS["dp"],
            minDist=DETECTION_PARAMS["minDist"],
            param1=DETECTION_PARAMS["param1"],
            param2=DETECTION_PARAMS["param2"],
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        print(f"Detected raw circles: {0 if circles is None else len(circles[0])}")

        if circles is not None and len(circles[0]) > 0:
            # show detected circles before filtering on a copy of result image
            # save it in Results/intermediate for reference
            intermediate_dir = Path(DIRECTORIES["intermediate_output"])
            if not intermediate_dir.exists():
                intermediate_dir.mkdir(parents=True, exist_ok=True)

            temp_result = self.result.copy()
            raw_circles = np.uint16(np.around(circles[0]))
            self.draw_circles(raw_circles, color=(255, 0, 0), thickness=3)

            # save image
            intermediate_path = (
                intermediate_dir / f"raw_circles_{Path(self.image_path).name}"
            )
            cv2.imwrite(str(intermediate_path), self.result)
            print(f"  Saved intermediate raw circles to: {intermediate_path}")

            # reset result image
            self.result = temp_result

        if circles is None:
            return []

        circles = np.uint16(np.around(circles[0]))

        # Filter circles
        valid_circles = []

        for circle in circles:
            x, y, radius = circle

            # Check if circle is complete (within image bounds)
            if not self.is_circle_complete(x, y, radius, mask):
                continue

            # Calculate quality score
            quality = self.calculate_circle_quality(x, y, radius)

            # Keep only high-quality circles
            if quality >= quality_threshold:
                print(
                    f"Accepted circle at ({x}, {y}) with radius {radius}, quality: {quality:.2f}"
                )
                valid_circles.append((int(x), int(y), int(radius), quality))

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
            cv2.putText(
                self.result,
                str(idx),
                (x - 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )

    def visualize_results(self, circles):
        """Display original and result side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original
        axes[0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Thermal Image")
        axes[0].axis("off")

        # Result
        axes[1].imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Detected Complete Circles: {len(circles)}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
