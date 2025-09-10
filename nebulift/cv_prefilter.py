"""
Computer Vision Pre-filter Module

Traditional computer vision techniques for detecting common astrophotography artifacts
like satellite streaks, airplane trails, and cloud contamination. This module provides
initial filtering to help with training data generation and reduces manual labeling effort.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, morphology

logger = logging.getLogger(__name__)


class ArtifactDetector:
    """Detects common astrophotography artifacts using traditional CV methods."""

    def __init__(self, min_streak_length: int = 50, min_streak_strength: float = 0.3):
        """
        Initialize artifact detector.
        
        Args:
            min_streak_length: Minimum length in pixels for streak detection
            min_streak_strength: Minimum strength threshold for streaks
        """
        self.min_streak_length = min_streak_length
        self.min_streak_strength = min_streak_strength

    def detect_linear_streaks(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect linear streaks (satellites, airplanes) using Hough transform.
        
        Args:
            image: Normalized image array [0, 1]
            
        Returns:
            Dictionary containing streak detection results
        """
        # Convert to uint8 for OpenCV operations
        image_uint8 = (image * 255).astype(np.uint8)

        # Apply edge detection
        edges = cv2.Canny(image_uint8, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=self.min_streak_length,
            maxLineGap=10,
        )

        streak_info = {
            "has_streaks": False,
            "num_streaks": 0,
            "streak_coordinates": [],
            "streak_angles": [],
            "streak_lengths": [],
            "max_streak_length": 0,
            "streak_density": 0.0,
        }

        if lines is not None:
            streak_info["has_streaks"] = True
            streak_info["num_streaks"] = len(lines)

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate streak properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                streak_info["streak_coordinates"].append([(x1, y1), (x2, y2)])
                streak_info["streak_angles"].append(angle)
                streak_info["streak_lengths"].append(length)

            if streak_info["streak_lengths"]:  # Type guard
                lengths = streak_info["streak_lengths"]
                assert isinstance(lengths, list), "Expected list of lengths"
                streak_info["max_streak_length"] = max(lengths)
            streak_info["streak_density"] = len(lines) / (image.shape[0] * image.shape[1])

        return streak_info

    def detect_clouds(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect cloud contamination using texture analysis.
        
        Args:
            image: Normalized image array [0, 1]
            
        Returns:
            Dictionary containing cloud detection results
        """
        # Apply Gaussian filter to smooth the image
        smoothed = filters.gaussian(image, sigma=2.0)

        # Calculate local standard deviation (texture measure)
        local_std = ndimage.generic_filter(image, np.std, size=15)

        # Calculate gradient magnitude
        grad_x = ndimage.sobel(smoothed, axis=1)
        grad_y = ndimage.sobel(smoothed, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Detect large smooth regions (potential clouds)
        cloud_mask = (local_std < 0.05) & (gradient_magnitude < 0.1) & (smoothed > 0.1)

        # Remove small regions
        cloud_mask = morphology.remove_small_objects(cloud_mask, min_size=1000)

        # Calculate cloud properties
        labeled_clouds, num_clouds = ndimage.label(cloud_mask)

        cloud_info = {
            "has_clouds": num_clouds > 0,
            "num_cloud_regions": num_clouds,
            "cloud_coverage_percent": np.sum(cloud_mask) / cloud_mask.size * 100,
            "cloud_mask": cloud_mask,
            "largest_cloud_area": 0,
        }

        if num_clouds > 0:
            cloud_areas = [np.sum(labeled_clouds == i) for i in range(1, num_clouds + 1)]
            cloud_info["largest_cloud_area"] = max(cloud_areas)

        return cloud_info

    def detect_saturation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect saturated regions that might indicate overexposure.
        
        Args:
            image: Normalized image array [0, 1]
            
        Returns:
            Dictionary containing saturation detection results
        """
        # Find saturated pixels (very bright)
        saturation_threshold = 0.95
        saturated_mask = image >= saturation_threshold

        # Find connected saturated regions
        labeled_sat, num_sat_regions = ndimage.label(saturated_mask)

        saturation_info = {
            "has_saturation": np.sum(saturated_mask) > 0,
            "saturation_percent": np.sum(saturated_mask) / saturated_mask.size * 100,
            "num_saturated_regions": num_sat_regions,
            "saturated_mask": saturated_mask,
        }

        return saturation_info

    def detect_hot_pixels(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect hot pixels and cosmic ray hits.
        
        Args:
            image: Normalized image array [0, 1]
            
        Returns:
            Dictionary containing hot pixel detection results
        """
        # Apply median filter to identify outliers
        median_filtered = ndimage.median_filter(image, size=3)

        # Find pixels significantly brighter than their neighborhood
        diff = image - median_filtered
        hot_pixel_threshold = 0.2
        hot_pixels = diff > hot_pixel_threshold

        # Remove large connected regions (these are likely real features)
        hot_pixels = morphology.remove_small_objects(hot_pixels, min_size=10)

        hot_pixel_info = {
            "has_hot_pixels": np.sum(hot_pixels) > 0,
            "num_hot_pixels": np.sum(hot_pixels),
            "hot_pixel_density": np.sum(hot_pixels) / hot_pixels.size,
            "hot_pixel_mask": hot_pixels,
        }

        return hot_pixel_info

    def calculate_image_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate various image quality metrics.
        
        Args:
            image: Normalized image array [0, 1]
            
        Returns:
            Dictionary of quality metrics
        """
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian((image * 255).astype(np.uint8), cv2.CV_64F)
        sharpness = laplacian.var()

        # Calculate contrast using standard deviation
        contrast = np.std(image)

        # Calculate noise estimate using high-frequency content
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = ndimage.convolve(image, kernel)
        noise_estimate = np.std(high_freq)

        # Calculate signal-to-noise ratio estimate
        signal_estimate = np.std(filters.gaussian(image, sigma=2))
        snr_estimate = signal_estimate / (noise_estimate + 1e-10)

        # Calculate star density (approximate using peak detection)
        # Apply star enhancement filter
        star_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = ndimage.convolve(image, star_kernel)

        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_maxima = (enhanced == maximum_filter(enhanced, size=5)) & (enhanced > 0.5)
        star_count = np.sum(local_maxima)

        return {
            "sharpness": float(sharpness),
            "contrast": float(contrast),
            "noise_estimate": float(noise_estimate),
            "snr_estimate": float(snr_estimate),
            "estimated_star_count": int(star_count),
            "star_density": float(star_count / image.size),
        }

    def comprehensive_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive artifact and quality analysis.
        
        Args:
            image: Normalized image array [0, 1]
            
        Returns:
            Complete analysis results
        """
        results: Dict[str, Any] = {
            "streaks": self.detect_linear_streaks(image),
            "clouds": self.detect_clouds(image),
            "saturation": self.detect_saturation(image),
            "hot_pixels": self.detect_hot_pixels(image),
            "quality_metrics": self.calculate_image_quality_metrics(image),
        }

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(results)
        results["overall_quality_score"] = quality_score

        # Determine if image should be flagged for manual review
        results["needs_manual_review"] = self._needs_manual_review(results)

        return results

    def _calculate_quality_score(self, analysis_results: Dict[str, Any]) -> float:
        """
        Calculate overall quality score from analysis results.
        
        Args:
            analysis_results: Results from comprehensive_analysis
            
        Returns:
            Quality score between 0 (poor) and 1 (excellent)
        """
        score = 1.0

        # Penalize for streaks
        if analysis_results["streaks"]["has_streaks"]:
            streak_penalty = min(0.5, analysis_results["streaks"]["num_streaks"] * 0.1)
            score -= streak_penalty

        # Penalize for clouds
        if analysis_results["clouds"]["has_clouds"]:
            cloud_penalty = min(0.4, analysis_results["clouds"]["cloud_coverage_percent"] * 0.01)
            score -= cloud_penalty

        # Penalize for excessive saturation
        if analysis_results["saturation"]["saturation_percent"] > 1.0:
            sat_penalty = min(0.3, analysis_results["saturation"]["saturation_percent"] * 0.01)
            score -= sat_penalty

        # Penalize for too many hot pixels
        if analysis_results["hot_pixels"]["hot_pixel_density"] > 0.001:
            hot_penalty = min(0.2, analysis_results["hot_pixels"]["hot_pixel_density"] * 100)
            score -= hot_penalty

        # Bonus for good sharpness and contrast
        metrics = analysis_results["quality_metrics"]
        if metrics["sharpness"] > 100:
            score += 0.1
        if metrics["contrast"] > 0.2:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _needs_manual_review(self, analysis_results: Dict[str, Any]) -> bool:
        """
        Determine if image needs manual review based on analysis.
        
        Args:
            analysis_results: Results from comprehensive_analysis
            
        Returns:
            True if manual review is recommended
        """
        # Flag for manual review if:
        # - Has streaks
        # - Significant cloud coverage
        # - Quality score is borderline

        if analysis_results["streaks"]["has_streaks"]:
            return True

        if analysis_results["clouds"]["cloud_coverage_percent"] > 5.0:
            return True

        if 0.3 < analysis_results["overall_quality_score"] < 0.7:
            return True

        return False


def batch_analyze_images(image_paths: List[str],
                        detector: Optional[ArtifactDetector] = None) -> Dict[str, Dict[str, Any]]:
    """
    Analyze a batch of images for artifacts and quality.
    
    Args:
        image_paths: List of paths to image files
        detector: Optional pre-configured detector instance
        
    Returns:
        Dictionary mapping file paths to analysis results
    """
    if detector is None:
        detector = ArtifactDetector()

    results = {}

    for image_path in image_paths:
        try:
            # Load image (this would use the FITSProcessor in practice)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Normalize to [0, 1]
                image_norm = image.astype(np.float64) / 255.0

                # Analyze
                analysis = detector.comprehensive_analysis(image_norm)
                results[image_path] = analysis
            else:
                logger.warning(f"Could not load image: {image_path}")

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")

    return results


def filter_images_by_quality(analysis_results: Dict[str, Dict[str, Any]],
                            quality_threshold: float = 0.6) -> Tuple[List[str], List[str]]:
    """
    Filter images based on quality analysis results.
    
    Args:
        analysis_results: Results from batch_analyze_images
        quality_threshold: Minimum quality score for acceptance
        
    Returns:
        Tuple of (good_images, poor_images) file paths
    """
    good_images = []
    poor_images = []

    for image_path, analysis in analysis_results.items():
        if analysis["overall_quality_score"] >= quality_threshold:
            good_images.append(image_path)
        else:
            poor_images.append(image_path)

    return good_images, poor_images
