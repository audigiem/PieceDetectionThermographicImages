"""
Main Batch Processor
Processes multiple thermal images for circle detection and generates comprehensive reports.
"""

import cv2
import numpy as np
from pathlib import Path
from circle_detector import ImprovedCircleDetector
from visualization import (
    create_combined_results_visualization,
    create_statistics_visualization,
    save_results_to_json,
    save_results_to_text,
)
from config import DETECTION_PARAMS, DIRECTORIES


def process_image(image_path, save=True, display=False):
    """
    Process a single image to detect complete circles.

    Args:
        image_path: Path to thermal image
        save: Whether to save result
        display: Whether to display result

    Returns:
        Tuple of (detector, circles)
    """
    print(f"\nProcessing: {Path(image_path).name}")

    detector = ImprovedCircleDetector(image_path)

    # Detect complete circles using config defaults
    circles = detector.detect_complete_circles()

    print(f"✓ Detected {len(circles)} complete circles")

    if len(circles) > 0:
        detector.draw_circles(circles)

        # Print details
        radii = [r for _, _, r in circles]
        print(f"  Mean radius: {np.mean(radii):.1f} px")
        print(f"  Radius range: [{min(radii)}, {max(radii)}] px")

    # Save result
    if save:
        output_dir = Path(DIRECTORIES["images_output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_path), detector.result)
        print(f"  Saved to: {output_path}")

    # Display
    if display:
        detector.visualize_results(circles)

    return detector, circles


def process_all_images(images_dir=None, save=True, generate_reports=True):
    """
    Process all images in directory and generate comprehensive reports.

    Args:
        images_dir: Directory containing input images (uses config default if None)
        save: Whether to save individual results
        generate_reports: Whether to generate combined visualizations and reports
    """
    if images_dir is None:
        images_dir = DIRECTORIES["input_dir"]

    images_path = Path(images_dir)
    image_files = sorted(images_path.glob("*.png"))
    image_files.extend(sorted(images_path.glob("*.jpg")))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print("=" * 70)
    print("IMPROVED CIRCLE DETECTION - COMPLETE CIRCLES ONLY")
    print("=" * 70)
    print(f"\nProcessing {len(image_files)} images...\n")

    results = []

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}]", end=" ")
        try:
            detector, circles = process_image(img_path, save=save, display=False)
            results.append(
                {
                    "filename": img_path.name,
                    "count": len(circles),
                    "circles": circles,
                    "detector": detector,
                }
            )
        except Exception as e:
            print(f"  Error: {e}")
            results.append(
                {
                    "filename": img_path.name,
                    "count": 0,
                    "circles": [],
                    "detector": None,
                    "error": str(e),
                }
            )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for result in results:
        status = "✓" if result.get("detector") is not None else "✗"
        print(f"{status} {result['filename']:20} -> {result['count']} complete circles")

    total_circles = sum(r["count"] for r in results)
    print(f"\nTotal circles detected: {total_circles}")

    # Generate comprehensive reports
    if generate_reports:
        print("\n" + "=" * 70)
        print("GENERATING REPORTS")
        print("=" * 70)

        # Create combined visualization
        create_combined_results_visualization(results)

        # Create statistics visualization
        create_statistics_visualization(results)

        # Save JSON report
        save_results_to_json(results)

        # Save text report
        save_results_to_text(results)

        print("\n" + "=" * 70)
        print("ALL REPORTS GENERATED SUCCESSFULLY")
        print("=" * 70)
        print("\nOutput files:")
        print(
            "  - Results/visualizations/all_results.png      (Combined detection results)"
        )
        print(
            "  - Results/visualizations/circle_counts.png    (Bar chart of detections)"
        )
        print("  - Results/visualizations/radius_distribution.png (Histogram)")
        print("  - Results/visualizations/radius_boxplot.png   (Box plot by image)")
        print("  - Results/visualizations/summary_table.png    (Summary statistics)")
        print(
            "  - Results/detection_report.json               (Machine-readable report)"
        )
        print("  - Results/detection_report.txt                (Human-readable report)")
        print(
            "  - Results/images/                             (Individual detection images)"
        )
        print("  - Results/masks/                              (ROI masks)")
        print("  - Results/intermediate/                       (Raw circle detections)")
        print("  - Results/rejected_circles/                   (Filtered out circles)")


# Example usage
if __name__ == "__main__":
    # Process single image with visualization
    # process_image('Images/image2.png', save=False, display=True)

    # Process all images and generate comprehensive reports
    process_all_images("Images", save=True, generate_reports=True)
