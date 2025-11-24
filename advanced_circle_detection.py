"""
Advanced Circle Detection with Multiple Techniques
This script provides enhanced methods for detecting circles in thermal images
including edge-based detection and template matching.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


class AdvancedCircleDetector:
    """Advanced circle detection with multiple techniques."""

    def __init__(self, image_path):
        """Initialize with image path."""
        self.image_path = image_path
        self.original = cv2.imread(str(image_path))
        if self.original is None:
            raise ValueError(f"Cannot load image: {image_path}")

        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.result = self.original.copy()

    def enhance_thermal_image(self):
        """
        Enhanced preprocessing specifically for thermal images.

        Returns:
            Enhanced grayscale image
        """
        # Bilateral filter preserves edges while reducing noise
        bilateral = cv2.bilateralFilter(self.gray, 9, 75, 75)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)

        # Optional: Enhance hot spots (bright regions) typical in thermal images
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def detect_with_canny_and_fitting(self, low_threshold=50, high_threshold=150):
        """
        Detect circles using Canny edge detection and circle fitting.

        Args:
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny

        Returns:
            List of detected circles
        """
        enhanced = self.enhance_thermal_image()

        # Detect edges
        edges = cv2.Canny(enhanced, low_threshold, high_threshold)

        # Dilate edges slightly to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        for contour in contours:
            # Need at least 5 points to fit a circle
            if len(contour) < 5:
                continue

            # Fit ellipse and check if it's circular
            try:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse

                # Check if it's approximately circular (aspect ratio close to 1)
                aspect_ratio = max(MA, ma) / (min(MA, ma) + 1e-6)

                if aspect_ratio < 1.3 and 10 < ma < 200:  # Circular enough
                    radius = int((MA + ma) / 4)
                    circles.append((int(x), int(y), radius))
            except:
                continue

        # Remove overlapping circles (non-maximum suppression)
        circles = self._non_maximum_suppression(circles)

        return circles

    def detect_with_blob_detector(self):
        """
        Detect circles using SimpleBlobDetector.

        Returns:
            List of detected circles
        """
        enhanced = self.enhance_thermal_image()

        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Area
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 50000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.7

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(enhanced)

        # Convert to circles format
        circles = [(int(kp.pt[0]), int(kp.pt[1]), int(kp.size / 2))
                   for kp in keypoints]

        return circles

    def detect_with_watershed(self):
        """
        Detect circles using watershed segmentation.

        Returns:
            List of detected circles
        """
        enhanced = self.enhance_thermal_image()

        # Threshold
        _, binary = cv2.threshold(enhanced, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(),
                                   255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(self.original, markers)

        # Extract circles from markers
        circles = []
        for label in range(2, markers.max() + 1):
            mask = np.zeros_like(self.gray)
            mask[markers == label] = 255

            # Find contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area > 100:
                    (x, y), radius = cv2.minEnclosingCircle(contour)

                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.6:
                            circles.append((int(x), int(y), int(radius)))

        return circles

    def _non_maximum_suppression(self, circles, overlap_threshold=0.5):
        """
        Remove overlapping circles using non-maximum suppression.

        Args:
            circles: List of (x, y, radius) tuples
            overlap_threshold: Overlap threshold for suppression

        Returns:
            Filtered list of circles
        """
        if len(circles) == 0:
            return []

        circles = np.array(circles)

        # Sort by radius (prefer larger circles)
        sorted_indices = np.argsort(circles[:, 2])[::-1]

        keep = []
        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            keep.append(current)

            if len(sorted_indices) == 1:
                break

            # Calculate distances to remaining circles
            current_circle = circles[current]
            remaining_circles = circles[sorted_indices[1:]]

            distances = np.sqrt(
                (remaining_circles[:, 0] - current_circle[0])**2 +
                (remaining_circles[:, 1] - current_circle[1])**2
            )

            # Remove overlapping circles
            min_distance = current_circle[2] + remaining_circles[:, 2]
            non_overlapping = distances > min_distance * overlap_threshold

            sorted_indices = sorted_indices[1:][non_overlapping]

        return circles[keep].tolist()

    def ensemble_detection(self, methods=['hough', 'canny', 'blob']):
        """
        Combine multiple detection methods for robust detection.

        Args:
            methods: List of methods to use

        Returns:
            Combined list of detected circles
        """
        all_circles = []

        if 'hough' in methods:
            # Hough circles
            enhanced = self.enhance_thermal_image()
            hough = cv2.HoughCircles(
                enhanced, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=25, minRadius=5, maxRadius=200
            )
            if hough is not None:
                for circle in hough[0]:
                    all_circles.append((int(circle[0]), int(circle[1]), int(circle[2])))

        if 'canny' in methods:
            canny_circles = self.detect_with_canny_and_fitting()
            all_circles.extend(canny_circles)

        if 'blob' in methods:
            blob_circles = self.detect_with_blob_detector()
            all_circles.extend(blob_circles)

        if 'watershed' in methods:
            watershed_circles = self.detect_with_watershed()
            all_circles.extend(watershed_circles)

        # Apply clustering to merge similar detections
        if len(all_circles) > 0:
            circles = self._cluster_circles(all_circles)
        else:
            circles = []

        return circles

    def _cluster_circles(self, circles, distance_threshold=20):
        """
        Cluster similar circle detections.

        Args:
            circles: List of detected circles
            distance_threshold: Distance threshold for clustering

        Returns:
            Merged circles
        """
        if len(circles) == 0:
            return []

        circles = np.array(circles)

        # Simple clustering: merge circles with centers close together
        merged = []
        used = set()

        for i in range(len(circles)):
            if i in used:
                continue

            cluster = [circles[i]]
            used.add(i)

            for j in range(i + 1, len(circles)):
                if j in used:
                    continue

                dist = np.sqrt(
                    (circles[i][0] - circles[j][0])**2 +
                    (circles[i][1] - circles[j][1])**2
                )

                if dist < distance_threshold:
                    cluster.append(circles[j])
                    used.add(j)

            # Average the cluster
            cluster = np.array(cluster)
            avg_circle = (
                int(np.mean(cluster[:, 0])),
                int(np.mean(cluster[:, 1])),
                int(np.mean(cluster[:, 2]))
            )
            merged.append(avg_circle)

        return merged

    def draw_circles(self, circles, color=(0, 255, 0), thickness=2):
        """Draw circles on result image."""
        for (x, y, radius) in circles:
            cv2.circle(self.result, (x, y), radius, color, thickness)
            cv2.circle(self.result, (x, y), 3, (0, 0, 255), -1)

    def visualize_processing_steps(self):
        """Visualize intermediate processing steps."""
        enhanced = self.enhance_thermal_image()
        edges = cv2.Canny(enhanced, 50, 150)

        _, binary = cv2.threshold(enhanced, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(self.gray, cmap='gray')
        axes[0, 0].set_title('Original Grayscale')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title('Binary Threshold')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()


def interactive_parameter_tuning(image_path):
    """
    Interactive tool for tuning circle detection parameters.

    Args:
        image_path: Path to image
    """
    detector = AdvancedCircleDetector(image_path)

    print("\n" + "="*60)
    print("INTERACTIVE PARAMETER TUNING")
    print("="*60)
    print("\nTesting different parameter combinations...\n")

    # Test different parameter combinations
    param_sets = [
        {"min_dist": 30, "param1": 50, "param2": 25, "min_r": 5, "max_r": 200},
        {"min_dist": 50, "param1": 100, "param2": 30, "min_r": 10, "max_r": 150},
        {"min_dist": 20, "param1": 30, "param2": 20, "min_r": 5, "max_r": 250},
    ]

    enhanced = detector.enhance_thermal_image()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(detector.original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    for idx, params in enumerate(param_sets):
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT, dp=1,
            minDist=params["min_dist"],
            param1=params["param1"],
            param2=params["param2"],
            minRadius=params["min_r"],
            maxRadius=params["max_r"]
        )

        result = detector.original.copy()
        count = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            count = len(circles[0])
            for circle in circles[0]:
                cv2.circle(result, (circle[0], circle[1]), circle[2],
                          (0, 255, 0), 2)
                cv2.circle(result, (circle[0], circle[1]), 3, (0, 0, 255), -1)

        row = (idx + 1) // 2
        col = (idx + 1) % 2
        axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'Config {idx+1}: {count} circles\n' +
                                 f'minDist={params["min_dist"]}, param2={params["param2"]}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    image_path = "Images/image1.png"

    if Path(image_path).exists():
        print("Advanced Circle Detection Demo")
        print("="*60)

        # Create detector
        detector = AdvancedCircleDetector(image_path)

        # Visualize processing steps
        print("\n1. Visualizing processing steps...")
        detector.visualize_processing_steps()

        # Ensemble detection
        print("\n2. Running ensemble detection...")
        circles = detector.ensemble_detection(methods=['hough', 'canny', 'blob'])
        print(f"   Detected {len(circles)} circles")

        detector.draw_circles(circles)

        # Display result
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(detector.original, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(detector.result, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Circles ({len(circles)})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Interactive parameter tuning
        print("\n3. Interactive parameter tuning...")
        interactive_parameter_tuning(image_path)
    else:
        print(f"Image not found: {image_path}")
        print("Please ensure the Images directory exists with image files.")

