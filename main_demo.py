"""
Main Demo Script - Circle Detection in Thermal Images
This script demonstrates all capabilities of the circle detection system.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from circle_detection import CircleDetector, process_single_image, compare_methods
from advanced_circle_detection import AdvancedCircleDetector, interactive_parameter_tuning
from batch_processor import BatchProcessor


def print_menu():
    """Display the main menu."""
    print("\n" + "="*70)
    print("CIRCLE DETECTION IN THERMAL IMAGES - MAIN DEMO")
    print("="*70)
    print("\nSelect an option:")
    print("\n1. Process ALL images (batch mode) - RECOMMENDED")
    print("2. Process a SINGLE image (interactive)")
    print("3. Compare detection methods on one image")
    print("4. Interactive parameter tuning")
    print("5. Advanced ensemble detection demo")
    print("6. Show processing steps visualization")
    print("7. Run quick test on sample image")
    print("\n0. Exit")
    print("\n" + "="*70)


def option1_batch_processing():
    """Batch process all images."""
    print("\n" + "="*70)
    print("BATCH PROCESSING MODE")
    print("="*70)

    input_dir = "Images"
    output_dir = "Results"

    if not Path(input_dir).exists():
        print(f"\n✗ Error: '{input_dir}' directory not found!")
        print("Please ensure the Images directory exists with thermal images.")
        return

    image_files = list(Path(input_dir).glob('*.png'))
    image_files.extend(list(Path(input_dir).glob('*.jpg')))

    if not image_files:
        print(f"\n✗ No images found in '{input_dir}' directory!")
        return

    print(f"\nFound {len(image_files)} images in '{input_dir}'")
    print("\nProcessing options:")
    print("  1. Auto (system chooses best method)")
    print("  2. Hough Transform only")
    print("  3. Advanced ensemble method")

    choice = input("\nSelect method (1-3, default=1): ").strip() or "1"

    method_map = {'1': 'auto', '2': 'hough', '3': 'advanced'}
    method = method_map.get(choice, 'auto')

    processor = BatchProcessor(input_dir, output_dir)
    processor.process_all_images(method=method)

    print("\n✓ Batch processing complete!")
    print(f"✓ Results saved to '{output_dir}' directory")


def option2_single_image():
    """Process a single image interactively."""
    print("\n" + "="*70)
    print("SINGLE IMAGE PROCESSING")
    print("="*70)

    # List available images
    input_dir = Path("Images")
    if not input_dir.exists():
        print(f"\n✗ Error: 'Images' directory not found!")
        return

    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))

    if not image_files:
        print("\n✗ No images found!")
        return

    print("\nAvailable images:")
    for idx, img_file in enumerate(image_files, 1):
        print(f"  {idx}. {img_file.name}")

    choice = input(f"\nSelect image (1-{len(image_files)}): ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(image_files):
            image_path = image_files[idx]

            print("\nDetection methods:")
            print("  1. Hough Circle Transform")
            print("  2. Contour-based detection")
            print("  3. Advanced ensemble")

            method_choice = input("\nSelect method (1-3, default=1): ").strip() or "1"

            if method_choice in ['1', '2']:
                method = 'hough' if method_choice == '1' else 'contours'
                process_single_image(str(image_path), method=method, display=True, save=True)
            else:
                detector = AdvancedCircleDetector(str(image_path))
                circles = detector.ensemble_detection(methods=['hough', 'canny', 'blob'])
                detector.draw_circles(circles)

                print(f"\n✓ Detected {len(circles)} circles")

                # Display
                plt.figure(figsize=(12, 6))
                plt.subplot(121)
                plt.imshow(cv2.cvtColor(detector.original, cv2.COLOR_BGR2RGB))
                plt.title('Original')
                plt.axis('off')

                plt.subplot(122)
                plt.imshow(cv2.cvtColor(detector.result, cv2.COLOR_BGR2RGB))
                plt.title(f'Detected: {len(circles)} circles')
                plt.axis('off')

                plt.tight_layout()
                plt.show()
        else:
            print("Invalid selection!")
    except ValueError:
        print("Invalid input!")


def option3_compare_methods():
    """Compare detection methods."""
    print("\n" + "="*70)
    print("METHOD COMPARISON")
    print("="*70)

    input_dir = Path("Images")
    if not input_dir.exists():
        print("\n✗ Error: 'Images' directory not found!")
        return

    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))

    if not image_files:
        print("\n✗ No images found!")
        return

    print("\nAvailable images:")
    for idx, img_file in enumerate(image_files, 1):
        print(f"  {idx}. {img_file.name}")

    choice = input(f"\nSelect image (1-{len(image_files)}): ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(image_files):
            image_path = image_files[idx]
            compare_methods(str(image_path))
        else:
            print("Invalid selection!")
    except ValueError:
        print("Invalid input!")


def option4_parameter_tuning():
    """Interactive parameter tuning."""
    print("\n" + "="*70)
    print("PARAMETER TUNING")
    print("="*70)

    input_dir = Path("Images")
    if not input_dir.exists():
        print("\n✗ Error: 'Images' directory not found!")
        return

    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))

    if not image_files:
        print("\n✗ No images found!")
        return

    print("\nAvailable images:")
    for idx, img_file in enumerate(image_files, 1):
        print(f"  {idx}. {img_file.name}")

    choice = input(f"\nSelect image (1-{len(image_files)}): ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(image_files):
            image_path = image_files[idx]
            interactive_parameter_tuning(str(image_path))
        else:
            print("Invalid selection!")
    except ValueError:
        print("Invalid input!")


def option5_ensemble_demo():
    """Advanced ensemble detection demo."""
    print("\n" + "="*70)
    print("ENSEMBLE DETECTION DEMO")
    print("="*70)

    input_dir = Path("Images")
    if not input_dir.exists():
        print("\n✗ Error: 'Images' directory not found!")
        return

    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))

    if not image_files:
        print("\n✗ No images found!")
        return

    # Use first image for demo
    image_path = image_files[0]
    print(f"\nDemonstrating on: {image_path.name}")

    detector = AdvancedCircleDetector(str(image_path))

    # Try different method combinations
    methods_list = [
        ['hough'],
        ['canny'],
        ['blob'],
        ['hough', 'canny', 'blob']
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, methods in enumerate(methods_list):
        det = AdvancedCircleDetector(str(image_path))
        circles = det.ensemble_detection(methods=methods)
        det.draw_circles(circles)

        axes[idx].imshow(cv2.cvtColor(det.result, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"Methods: {', '.join(methods)}\nDetected: {len(circles)} circles")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Demo complete!")


def option6_processing_steps():
    """Show processing steps visualization."""
    print("\n" + "="*70)
    print("PROCESSING STEPS VISUALIZATION")
    print("="*70)

    input_dir = Path("Images")
    if not input_dir.exists():
        print("\n✗ Error: 'Images' directory not found!")
        return

    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))

    if not image_files:
        print("\n✗ No images found!")
        return

    image_path = image_files[0]
    print(f"\nVisualizing processing steps for: {image_path.name}")

    detector = AdvancedCircleDetector(str(image_path))
    detector.visualize_processing_steps()

    print("\n✓ Visualization complete!")


def option7_quick_test():
    """Run a quick test on sample image."""
    print("\n" + "="*70)
    print("QUICK TEST")
    print("="*70)

    input_dir = Path("Images")
    if not input_dir.exists():
        print("\n✗ Error: 'Images' directory not found!")
        return

    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))

    if not image_files:
        print("\n✗ No images found!")
        return

    # Test on first image
    image_path = image_files[0]
    print(f"\nTesting on: {image_path.name}")

    print("\n1. Basic Hough detection...")
    detector1 = CircleDetector(str(image_path))
    circles1 = detector1.detect_circles_hough()

    if circles1 is not None:
        count1 = len(circles1[0]) if len(circles1.shape) == 3 else len(circles1)
        print(f"   ✓ Found {count1} circles")
    else:
        print("   ✗ No circles detected")
        count1 = 0

    print("\n2. Advanced ensemble detection...")
    detector2 = AdvancedCircleDetector(str(image_path))
    circles2 = detector2.ensemble_detection()
    count2 = len(circles2)
    print(f"   ✓ Found {count2} circles")

    # Quick visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(detector1.original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    if circles1 is not None:
        detector1.draw_circles(circles1)
    axes[1].imshow(cv2.cvtColor(detector1.result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Hough: {count1} circles')
    axes[1].axis('off')

    detector2.draw_circles(circles2)
    axes[2].imshow(cv2.cvtColor(detector2.result, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Ensemble: {count2} circles')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n✓ Quick test complete!")


def main():
    """Main entry point."""
    while True:
        print_menu()
        choice = input("\nEnter your choice (0-7): ").strip()

        if choice == '0':
            print("\nExiting. Goodbye!")
            break
        elif choice == '1':
            option1_batch_processing()
        elif choice == '2':
            option2_single_image()
        elif choice == '3':
            option3_compare_methods()
        elif choice == '4':
            option4_parameter_tuning()
        elif choice == '5':
            option5_ensemble_demo()
        elif choice == '6':
            option6_processing_steps()
        elif choice == '7':
            option7_quick_test()
        else:
            print("\n✗ Invalid choice! Please select 0-7.")

        if choice != '0':
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()

