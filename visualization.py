"""
Visualization Module
Creates comprehensive visualizations and statistics for circle detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def create_combined_results_visualization(
    results, output_path="Results/visualizations/all_results.png"
):
    """
    Create a single image showing all detection results in a grid.

    Args:
        results: List of detection results from batch processing
        output_path: Path to save the combined visualization
    """
    # Filter out results with errors
    valid_results = [r for r in results if r.get("detector") is not None]

    if not valid_results:
        print("No valid results to visualize")
        return

    n_images = len(valid_results)

    # Calculate grid dimensions (prefer wider grids)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

    # Handle single image case
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_images > 1 else axes

    for idx, result in enumerate(valid_results):
        detector = result["detector"]
        circles = result["circles"]
        filename = result["filename"]

        # Show result image with detections
        axes[idx].imshow(cv2.cvtColor(detector.result, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(
            f"{filename}\n{len(circles)} circles detected",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Saved combined results visualization to: {output_path}")


def create_statistics_visualization(results, output_dir="Results/visualizations"):
    """
    Create statistical visualizations for circle detection results.
    Creates multiple separate files to avoid overlapping.

    Args:
        results: List of detection results from batch processing
        output_dir: Directory to save the statistics visualizations
    """
    # Filter out results with errors
    valid_results = [r for r in results if r.get("detector") is not None]

    if not valid_results:
        print("No valid results for statistics")
        return

    # Extract statistics
    filenames = [r["filename"] for r in valid_results]
    circle_counts = [r["count"] for r in valid_results]
    all_radii = []
    radius_by_image = []

    for result in valid_results:
        radii = [r for _, _, r in result["circles"]]
        all_radii.extend(radii)
        radius_by_image.append(radii if radii else [0])

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(filenames)))

    # 1. Circle count per image (bar chart)
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    bars = ax1.bar(
        range(len(filenames)),
        circle_counts,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_xlabel("Image", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Circles", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Circle Detection Count per Image", fontsize=14, fontweight="bold", pad=20
    )
    ax1.set_xticks(range(len(filenames)))
    ax1.set_xticklabels(filenames, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, count in zip(bars, circle_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/circle_counts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved circle counts chart to: {output_dir}/circle_counts.png")

    # 2. Radius distribution histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if all_radii:
        ax2.hist(all_radii, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        ax2.axvline(
            np.mean(all_radii),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(all_radii):.1f}px",
        )
        ax2.axvline(
            np.median(all_radii),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(all_radii):.1f}px",
        )
        ax2.legend(fontsize=11)
    ax2.set_xlabel("Radius (pixels)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax2.set_title("Overall Radius Distribution", fontsize=14, fontweight="bold", pad=20)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/radius_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved radius distribution to: {output_dir}/radius_distribution.png")

    # 3. Box plot of radii by image
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    if all_radii:
        bp = ax3.boxplot(radius_by_image, labels=filenames, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax3.set_xlabel("Image", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Radius (pixels)", fontsize=12, fontweight="bold")
    ax3.set_title(
        "Radius Distribution by Image", fontsize=14, fontweight="bold", pad=20
    )
    ax3.set_xticklabels(filenames, rotation=45, ha="right")
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/radius_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved radius boxplot to: {output_dir}/radius_boxplot.png")

    # 4. Summary statistics table
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.axis("tight")
    ax4.axis("off")

    total_circles = sum(circle_counts)
    avg_circles = np.mean(circle_counts)

    summary_data = [
        ["Metric", "Value"],
        ["Total Images Processed", str(len(valid_results))],
        ["Total Circles Detected", str(total_circles)],
        ["Average Circles per Image", f"{avg_circles:.2f}"],
        ["Min Circles per Image", str(min(circle_counts))],
        ["Max Circles per Image", str(max(circle_counts))],
    ]

    if all_radii:
        summary_data.extend(
            [
                ["Average Radius", f"{np.mean(all_radii):.1f} px"],
                ["Median Radius", f"{np.median(all_radii):.1f} px"],
                ["Min Radius", f"{min(all_radii)} px"],
                ["Max Radius", f"{max(all_radii)} px"],
                ["Std Dev Radius", f"{np.std(all_radii):.1f} px"],
            ]
        )

    table = ax4.table(
        cellText=summary_data, cellLoc="left", loc="center", colWidths=[0.4, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)

    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax4.set_title("Summary Statistics", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved summary table to: {output_dir}/summary_table.png")


def save_results_to_json(results, output_path="Results/detection_report.json"):
    """
    Save detection results to JSON format.

    Args:
        results: List of detection results from batch processing
        output_path: Path to save the JSON file
    """
    # Prepare data for JSON (exclude detector objects)
    json_data = {
        "summary": {
            "total_images": len(results),
            "total_circles_detected": sum(r["count"] for r in results),
            "average_circles_per_image": (
                sum(r["count"] for r in results) / len(results) if results else 0
            ),
        },
        "images": [],
    }

    for result in results:
        image_data = {
            "filename": result["filename"],
            "circle_count": result["count"],
            "circles": [],
        }

        for idx, (x, y, radius) in enumerate(result["circles"], 1):
            circle_info = {
                "id": idx,
                "center_x": int(x),
                "center_y": int(y),
                "radius": int(radius),
                "area": float(np.pi * radius * radius),
            }
            image_data["circles"].append(circle_info)

        # Calculate statistics for this image
        if result["circles"]:
            radii = [r for _, _, r in result["circles"]]
            image_data["statistics"] = {
                "mean_radius": float(np.mean(radii)),
                "median_radius": float(np.median(radii)),
                "min_radius": int(min(radii)),
                "max_radius": int(max(radii)),
                "std_radius": float(np.std(radii)),
            }
        else:
            image_data["statistics"] = None

        json_data["images"].append(image_data)

    # Calculate overall statistics
    all_radii = [r for result in results for _, _, r in result["circles"]]
    if all_radii:
        json_data["summary"]["overall_statistics"] = {
            "mean_radius": float(np.mean(all_radii)),
            "median_radius": float(np.median(all_radii)),
            "min_radius": int(min(all_radii)),
            "max_radius": int(max(all_radii)),
            "std_radius": float(np.std(all_radii)),
        }

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ Saved JSON report to: {output_path}")


def save_results_to_text(results, output_path="Results/detection_report.txt"):
    """
    Save detection results to a human-readable text format.

    Args:
        results: List of detection results from batch processing
        output_path: Path to save the text file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CIRCLE DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary
        total_circles = sum(r["count"] for r in results)
        f.write(f"Total Images Processed: {len(results)}\n")
        f.write(f"Total Circles Detected: {total_circles}\n")
        f.write(f"Average Circles per Image: {total_circles / len(results):.2f}\n\n")

        # Overall statistics
        all_radii = [r for result in results for _, _, r in result["circles"]]
        if all_radii:
            f.write("Overall Radius Statistics:\n")
            f.write(f"  Mean: {np.mean(all_radii):.1f} px\n")
            f.write(f"  Median: {np.median(all_radii):.1f} px\n")
            f.write(f"  Min: {min(all_radii)} px\n")
            f.write(f"  Max: {max(all_radii)} px\n")
            f.write(f"  Std Dev: {np.std(all_radii):.1f} px\n\n")

        f.write("-" * 80 + "\n")
        f.write("DETAILED RESULTS BY IMAGE\n")
        f.write("-" * 80 + "\n\n")

        # Per-image details
        for result in results:
            f.write(f"Image: {result['filename']}\n")
            f.write(f"Circles Detected: {result['count']}\n")

            if result["circles"]:
                f.write("\nDetected Circles:\n")
                for idx, (x, y, radius) in enumerate(result["circles"], 1):
                    f.write(
                        f"  {idx}. Center: ({x}, {y}), Radius: {radius} px, Area: {np.pi * radius * radius:.1f} px²\n"
                    )

                radii = [r for _, _, r in result["circles"]]
                f.write(f"\nRadius Statistics:\n")
                f.write(f"  Mean: {np.mean(radii):.1f} px\n")
                f.write(f"  Range: [{min(radii)}, {max(radii)}] px\n")

            f.write("\n" + "-" * 80 + "\n\n")

    print(f"✓ Saved text report to: {output_path}")
