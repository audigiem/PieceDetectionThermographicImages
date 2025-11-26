"""
Configuration file for circle detection parameters.
Adjust these values based on your specific needs.
"""

# Detection Parameters
DETECTION_PARAMS = {
    # Circle size constraints
    "min_radius": 50,  # Minimum circle radius in pixels
    "max_radius": 200,  # Maximum circle radius in pixels
    # Quality thresholds
    "quality_threshold": 0.1,  # Minimum quality score (0-1)
    # Hough Circle Transform parameters
    "dp": 1,  # Inverse ratio of accumulator resolution
    "minDist": 90,  # Minimum distance between circle centers
    "param1": 100,  # Higher Canny threshold
    "param2": 40,  # Accumulator threshold for circle detection
    # Filtering parameters
    "margin": -10,  # Margin from image edges (negative = allow near edge)
    "roi_coverage": 0.95,  # Minimum ROI coverage ratio (0-1)
    "edge_samples": 36,  # Number of points sampled around perimeter
}

# Preprocessing Parameters
PREPROCESSING_PARAMS = {
    "gaussian_kernel": (15, 15),  # Gaussian blur kernel size
    "gaussian_sigma": 3,  # Gaussian blur sigma
    "clahe_clip_limit": 2.0,  # CLAHE contrast limit
    "clahe_tile_size": (8, 8),  # CLAHE tile grid size
}

# ROI Detection Parameters
ROI_PARAMS = {
    # HSV color range for warm colors
    "hsv_lower_1": (0, 100, 100),
    "hsv_upper_1": (30, 255, 255),
    "hsv_lower_2": (150, 100, 100),
    "hsv_upper_2": (180, 255, 255),
    # Percentile threshold
    "intensity_percentile": 70,  # Use top 30% warmest pixels
    # Morphological operations
    "morph_kernel_size": (7, 7),
    "fill_kernel_size": (15, 15),
}

# Output Parameters
OUTPUT_PARAMS = {
    "save_individual_results": True,
    "save_roi_masks": True,
    "save_intermediate_results": True,
    "save_rejected_circles": True,
    "generate_combined_visualization": True,
    "generate_statistics": True,
    "generate_json_report": True,
    "generate_text_report": True,
}

# Directory Configuration
DIRECTORIES = {
    "input_dir": "Images",
    "output_base": "Results",
    "images_output": "Results/images",
    "masks_output": "Results/masks",
    "intermediate_output": "Results/intermediate",
    "rejected_output": "Results/rejected_circles",
    "visualizations_output": "Results/visualizations",
}
