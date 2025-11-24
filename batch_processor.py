"""
Batch Processing Script for Circle Detection in Thermal Images
This script processes all images in the Images folder and generates comprehensive results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys

# Import our detection modules
from circle_detection import CircleDetector
from advanced_circle_detection import AdvancedCircleDetector


class BatchProcessor:
    """Batch process multiple thermal images for circle detection."""

    def __init__(self, input_dir, output_dir='Results'):
        """
        Initialize batch processor.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output results
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)

        self.results = []

    def process_all_images(self, method='auto'):
        """
        Process all images in the input directory.

        Args:
            method: Detection method ('hough', 'advanced', 'auto')
        """
        image_files = sorted(self.input_dir.glob('*.png'))
        image_files.extend(sorted(self.input_dir.glob('*.jpg')))

        if not image_files:
            print(f"No images found in {self.input_dir}")
            return

        print(f"\nProcessing {len(image_files)} images...")
        print("="*70)

        for idx, img_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")

            try:
                result = self.process_single_image(img_path, method)
                self.results.append(result)

                # Print summary for this image
                print(f"    ✓ Detected {result['circle_count']} circles")
                if result['circle_count'] > 0:
                    print(f"    ✓ Mean radius: {result['mean_radius']:.1f} px")
                    print(f"    ✓ Saved to: {result['output_path'].name}")

            except Exception as e:
                print(f"    ✗ Error: {e}")
                self.results.append({
                    'filename': img_path.name,
                    'status': 'error',
                    'error': str(e)
                })

        # Generate summary report
        self.generate_report()
        self.create_comparison_visualization()

        print("\n" + "="*70)
        print("Batch processing complete!")
        print(f"Results saved to: {self.output_dir}")

    def process_single_image(self, image_path, method='auto'):
        """
        Process a single image with the specified method.

        Args:
            image_path: Path to image
            method: Detection method

        Returns:
            Dictionary with results
        """
        if method == 'auto':
            # Try both methods and use the one with better results
            circles_hough = self._detect_hough(image_path)
            circles_advanced = self._detect_advanced(image_path)

            # Choose method with more reasonable detections
            if circles_hough is not None and circles_advanced is not None:
                if 0 < len(circles_advanced) < len(circles_hough):
                    method = 'advanced'
                else:
                    method = 'hough'
            elif circles_advanced is not None and len(circles_advanced) > 0:
                method = 'advanced'
            else:
                method = 'hough'

        # Detect with chosen method
        if method == 'hough':
            detector = CircleDetector(image_path)
            circles = detector.detect_circles_hough(
                min_dist=30, param1=50, param2=25,
                min_radius=5, max_radius=200
            )

            if circles is not None and len(circles.shape) == 3:
                circles_list = [(int(c[0]), int(c[1]), int(c[2]))
                               for c in circles[0]]
            else:
                circles_list = []

        else:  # advanced
            detector = AdvancedCircleDetector(image_path)
            circles_list = detector.ensemble_detection(
                methods=['hough', 'canny', 'blob']
            )

        # Draw circles
        if len(circles_list) > 0:
            detector.draw_circles(circles_list if method == 'hough' else circles_list)

            # Add annotations
            for idx, circle in enumerate(circles_list, 1):
                x, y = circle[0], circle[1]
                cv2.putText(detector.result_image if method == 'hough' else detector.result,
                           str(idx), (x - 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Save result
        output_path = self.output_dir / 'images' / f"detected_{image_path.name}"
        result_img = detector.result_image if method == 'hough' else detector.result
        cv2.imwrite(str(output_path), result_img)

        # Calculate statistics
        if len(circles_list) > 0:
            radii = [c[2] for c in circles_list]
            stats = {
                'circle_count': len(circles_list),
                'mean_radius': np.mean(radii),
                'std_radius': np.std(radii),
                'min_radius': np.min(radii),
                'max_radius': np.max(radii),
                'circles': circles_list
            }
        else:
            stats = {
                'circle_count': 0,
                'circles': []
            }

        result = {
            'filename': image_path.name,
            'method': method,
            'status': 'success',
            'output_path': output_path,
            **stats
        }

        return result

    def _detect_hough(self, image_path):
        """Helper to detect with Hough method."""
        try:
            detector = CircleDetector(image_path)
            return detector.detect_circles_hough()
        except:
            return None

    def _detect_advanced(self, image_path):
        """Helper to detect with advanced method."""
        try:
            detector = AdvancedCircleDetector(image_path)
            return detector.ensemble_detection()
        except:
            return None

    def generate_report(self):
        """Generate a comprehensive JSON report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(self.results),
            'successful': sum(1 for r in self.results if r.get('status') == 'success'),
            'failed': sum(1 for r in self.results if r.get('status') == 'error'),
            'total_circles_detected': sum(r.get('circle_count', 0) for r in self.results),
            'results': []
        }

        for result in self.results:
            if result.get('status') == 'success':
                # Remove circles list from report (too verbose)
                result_copy = result.copy()
                if 'circles' in result_copy:
                    result_copy['circles'] = len(result_copy['circles'])
                if 'output_path' in result_copy:
                    result_copy['output_path'] = str(result_copy['output_path'])
                report['results'].append(result_copy)
            else:
                report['results'].append(result)

        # Save report
        report_path = self.output_dir / 'detection_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved to: {report_path}")

        # Also create a human-readable text report
        self._generate_text_report(report)

    def _generate_text_report(self, report):
        """Generate human-readable text report."""
        report_path = self.output_dir / 'detection_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CIRCLE DETECTION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Total images processed: {report['total_images']}\n")
            f.write(f"Successful: {report['successful']}\n")
            f.write(f"Failed: {report['failed']}\n")
            f.write(f"Total circles detected: {report['total_circles_detected']}\n\n")

            f.write("="*70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*70 + "\n\n")

            for result in report['results']:
                if result.get('status') == 'success':
                    f.write(f"File: {result['filename']}\n")
                    f.write(f"  Method: {result['method']}\n")
                    f.write(f"  Circles detected: {result['circle_count']}\n")
                    if result['circle_count'] > 0:
                        f.write(f"  Mean radius: {result['mean_radius']:.2f} px\n")
                        f.write(f"  Std radius: {result['std_radius']:.2f} px\n")
                        f.write(f"  Radius range: [{result['min_radius']}, {result['max_radius']}] px\n")
                    f.write("\n")
                else:
                    f.write(f"File: {result['filename']}\n")
                    f.write(f"  Status: ERROR - {result.get('error', 'Unknown error')}\n\n")

        print(f"✓ Text report saved to: {report_path}")

    def create_comparison_visualization(self):
        """Create a comparison visualization of all results."""
        successful_results = [r for r in self.results if r.get('status') == 'success']

        if not successful_results:
            print("No successful results to visualize")
            return

        n_images = len(successful_results)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))

        if n_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, result in enumerate(successful_results):
            img_path = result['output_path']
            img = cv2.imread(str(img_path))

            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(f"{result['filename']}\n{result['circle_count']} circles")
                axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        viz_path = self.output_dir / 'visualizations' / 'all_results.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison visualization saved to: {viz_path}")

        # Show if requested
        # plt.show()
        plt.close()

        # Create summary statistics plot
        self._create_statistics_plot()

    def _create_statistics_plot(self):
        """Create statistical plots of detection results."""
        successful_results = [r for r in self.results
                            if r.get('status') == 'success' and r.get('circle_count', 0) > 0]

        if not successful_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Circles per image
        filenames = [r['filename'] for r in successful_results]
        counts = [r['circle_count'] for r in successful_results]

        axes[0, 0].bar(range(len(filenames)), counts, color='steelblue')
        axes[0, 0].set_xlabel('Image')
        axes[0, 0].set_ylabel('Number of Circles')
        axes[0, 0].set_title('Circles Detected per Image')
        axes[0, 0].set_xticks(range(len(filenames)))
        axes[0, 0].set_xticklabels(filenames, rotation=45, ha='right')

        # 2. Mean radius per image
        mean_radii = [r['mean_radius'] for r in successful_results]

        axes[0, 1].bar(range(len(filenames)), mean_radii, color='coral')
        axes[0, 1].set_xlabel('Image')
        axes[0, 1].set_ylabel('Mean Radius (px)')
        axes[0, 1].set_title('Mean Circle Radius per Image')
        axes[0, 1].set_xticks(range(len(filenames)))
        axes[0, 1].set_xticklabels(filenames, rotation=45, ha='right')

        # 3. Distribution of circle counts
        axes[1, 0].hist(counts, bins=max(counts), color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Number of Circles')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Circle Counts')

        # 4. Distribution of all radii
        all_radii = []
        for result in successful_results:
            for circle in result.get('circles', []):
                all_radii.append(circle[2])

        if all_radii:
            axes[1, 1].hist(all_radii, bins=20, color='plum', edgecolor='black')
            axes[1, 1].set_xlabel('Radius (px)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of All Circle Radii')

        plt.tight_layout()

        stats_path = self.output_dir / 'visualizations' / 'statistics.png'
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        print(f"✓ Statistics plot saved to: {stats_path}")
        plt.close()


def main():
    """Main entry point for batch processing."""
    print("\n" + "="*70)
    print("CIRCLE DETECTION IN THERMAL IMAGES - BATCH PROCESSOR")
    print("="*70)

    # Configuration
    input_dir = "Images"
    output_dir = "Results"

    # Check if input directory exists
    if not Path(input_dir).exists():
        print(f"\n✗ Error: Input directory '{input_dir}' not found!")
        print("Please ensure the Images directory exists with thermal images.")
        sys.exit(1)

    # Create processor and run
    processor = BatchProcessor(input_dir, output_dir)
    processor.process_all_images(method='auto')

    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nCheck the '{output_dir}' folder for:")
    print("  • Detected images with circles marked")
    print("  • JSON report with detailed statistics")
    print("  • Text report (human-readable)")
    print("  • Comparison visualizations")
    print("  • Statistical analysis plots")
    print()


if __name__ == "__main__":
    main()

