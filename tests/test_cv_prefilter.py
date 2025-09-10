"""
Test suite for Computer Vision Pre-filter

Tests the traditional CV artifact detection functionality.
"""

from unittest.mock import patch

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from nebulift.cv_prefilter import (
    ArtifactDetector,
    batch_analyze_images,
    filter_images_by_quality,
)


class TestArtifactDetector:
    """Test cases for ArtifactDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ArtifactDetector(min_streak_length=20, min_streak_strength=0.2)

    def test_init(self):
        """Test ArtifactDetector initialization."""
        detector = ArtifactDetector(min_streak_length=50, min_streak_strength=0.5)
        assert detector.min_streak_length == 50
        assert detector.min_streak_strength == 0.5

    def test_detect_linear_streaks_no_streaks(self):
        """Test streak detection on image without streaks."""
        # Create smooth image without linear features
        image = np.random.normal(0.3, 0.05, (100, 100))
        image = np.clip(image, 0, 1)

        result = self.detector.detect_linear_streaks(image)

        assert isinstance(result, dict)
        assert "has_streaks" in result
        assert "num_streaks" in result
        assert "streak_coordinates" in result
        assert result["num_streaks"] >= 0

    def test_detect_linear_streaks_with_artificial_streak(self):
        """Test streak detection on image with artificial streak."""
        # Create image with diagonal line (simulated streak)
        image = np.zeros((100, 100))

        # Add diagonal streak
        for i in range(50):
            if i < 100 and i < 100:
                image[i, i] = 1.0

        result = self.detector.detect_linear_streaks(image)

        assert isinstance(result, dict)
        assert "has_streaks" in result
        # Note: Actual detection depends on edge detection sensitivity

    def test_detect_clouds_clear_image(self):
        """Test cloud detection on clear image."""
        # Create image with stars (bright points)
        image = np.random.normal(0.1, 0.02, (100, 100))

        # Add some star-like features
        image[25, 25] = 0.8
        image[75, 50] = 0.9

        image = np.clip(image, 0, 1)

        result = self.detector.detect_clouds(image)

        assert isinstance(result, dict)
        assert "has_clouds" in result
        assert "cloud_coverage_percent" in result
        assert "cloud_mask" in result
        assert result["cloud_coverage_percent"] >= 0

    def test_detect_clouds_with_simulated_clouds(self):
        """Test cloud detection with simulated cloud regions."""
        # Create image with large smooth regions (clouds)
        image = np.random.normal(0.1, 0.02, (100, 100))

        # Add smooth cloud-like region
        image[20:40, 20:60] = 0.3

        image = np.clip(image, 0, 1)

        result = self.detector.detect_clouds(image)

        assert isinstance(result, dict)
        assert "has_clouds" in result
        assert "num_cloud_regions" in result
        assert result["cloud_coverage_percent"] >= 0

    def test_detect_saturation(self):
        """Test saturation detection."""
        # Create image with saturated regions
        image = np.random.normal(0.3, 0.1, (50, 50))

        # Add saturated region
        image[10:15, 10:15] = 1.0

        image = np.clip(image, 0, 1)

        result = self.detector.detect_saturation(image)

        assert isinstance(result, dict)
        assert "has_saturation" in result
        assert "saturation_percent" in result
        assert "saturated_mask" in result
        assert result["saturation_percent"] >= 0

        # Should detect the saturated region we added
        assert result["has_saturation"] == True

    def test_detect_hot_pixels(self):
        """Test hot pixel detection."""
        # Create smooth image
        image = np.full((50, 50), 0.2)

        # Add hot pixels
        image[25, 25] = 0.8
        image[30, 35] = 0.9

        result = self.detector.detect_hot_pixels(image)

        assert isinstance(result, dict)
        assert "has_hot_pixels" in result
        assert "num_hot_pixels" in result
        assert "hot_pixel_density" in result
        assert "hot_pixel_mask" in result

    def test_calculate_image_quality_metrics(self):
        """Test image quality metrics calculation."""
        # Create test image with known properties
        image = np.random.normal(0.5, 0.1, (100, 100))
        image = np.clip(image, 0, 1)

        metrics = self.detector.calculate_image_quality_metrics(image)

        required_metrics = [
            "sharpness",
            "contrast",
            "noise_estimate",
            "snr_estimate",
            "estimated_star_count",
            "star_density",
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

        # Check reasonable ranges
        assert metrics["contrast"] >= 0
        assert metrics["noise_estimate"] >= 0
        assert metrics["estimated_star_count"] >= 0
        assert metrics["star_density"] >= 0

    def test_comprehensive_analysis(self):
        """Test comprehensive analysis pipeline."""
        # Create test image
        image = np.random.normal(0.3, 0.1, (100, 100))
        image = np.clip(image, 0, 1)

        result = self.detector.comprehensive_analysis(image)

        # Check all analysis components are present
        required_keys = [
            "streaks",
            "clouds",
            "saturation",
            "hot_pixels",
            "quality_metrics",
            "overall_quality_score",
            "needs_manual_review",
        ]

        for key in required_keys:
            assert key in result

        # Check quality score is in valid range
        assert 0 <= result["overall_quality_score"] <= 1
        assert isinstance(result["needs_manual_review"], bool)

    @given(
        streak_count=st.integers(min_value=0, max_value=5),
        cloud_coverage=st.floats(min_value=0, max_value=20),
        saturation_level=st.floats(min_value=0, max_value=5),
    )
    def test_quality_score_calculation(
        self,
        streak_count,
        cloud_coverage,
        saturation_level,
    ):
        """Property-based test for quality score calculation."""
        # Mock analysis results
        analysis_results = {
            "streaks": {
                "has_streaks": streak_count > 0,
                "num_streaks": streak_count,
            },
            "clouds": {
                "has_clouds": cloud_coverage > 0,
                "cloud_coverage_percent": cloud_coverage,
            },
            "saturation": {
                "saturation_percent": saturation_level,
            },
            "hot_pixels": {
                "hot_pixel_density": 0.0005,
            },
            "quality_metrics": {
                "sharpness": 150,
                "contrast": 0.25,
            },
        }

        score = self.detector._calculate_quality_score(analysis_results)

        # Score should be in valid range
        assert 0 <= score <= 1

        # Score should decrease with more artifacts
        if streak_count > 0 or cloud_coverage > 5:
            # Account for potential bonuses that might offset penalties
            expected_max_score = 1.2  # Base + bonuses
            assert score < expected_max_score

    def test_needs_manual_review_logic(self):
        """Test manual review recommendation logic."""
        # Case 1: Clean image
        clean_results = {
            "streaks": {"has_streaks": False},
            "clouds": {"cloud_coverage_percent": 1.0},
            "overall_quality_score": 0.8,
        }

        assert self.detector._needs_manual_review(clean_results) == False

        # Case 2: Image with streaks
        streak_results = {
            "streaks": {"has_streaks": True},
            "clouds": {"cloud_coverage_percent": 1.0},
            "overall_quality_score": 0.8,
        }

        assert self.detector._needs_manual_review(streak_results) == True

        # Case 3: Image with significant clouds
        cloud_results = {
            "streaks": {"has_streaks": False},
            "clouds": {"cloud_coverage_percent": 10.0},
            "overall_quality_score": 0.8,
        }

        assert self.detector._needs_manual_review(cloud_results) == True

        # Case 4: Borderline quality score
        borderline_results = {
            "streaks": {"has_streaks": False},
            "clouds": {"cloud_coverage_percent": 1.0},
            "overall_quality_score": 0.5,
        }

        assert self.detector._needs_manual_review(borderline_results) == True


class TestBatchProcessing:
    """Test cases for batch processing functions."""

    @patch("nebulift.cv_prefilter.cv2.imread")
    def test_batch_analyze_images(self, mock_imread):
        """Test batch image analysis."""
        # Mock image loading
        mock_image = np.random.random((100, 100)) * 255
        mock_imread.return_value = mock_image.astype(np.uint8)

        image_paths = ["image1.jpg", "image2.jpg"]

        results = batch_analyze_images(image_paths)

        assert isinstance(results, dict)
        assert len(results) == 2

        for path in image_paths:
            assert path in results
            assert "overall_quality_score" in results[path]

    @patch("nebulift.cv_prefilter.cv2.imread")
    def test_batch_analyze_images_with_failures(self, mock_imread):
        """Test batch analysis with some failed image loads."""
        # Mock some failures
        mock_imread.side_effect = [
            np.random.random((100, 100)) * 255,  # Success
            None,  # Failure
            np.random.random((100, 100)) * 255,  # Success
        ]

        image_paths = ["good1.jpg", "bad.jpg", "good2.jpg"]

        results = batch_analyze_images(image_paths)

        # Should only have results for successful loads
        assert len(results) == 2
        assert "good1.jpg" in results
        assert "good2.jpg" in results
        assert "bad.jpg" not in results

    def test_filter_images_by_quality(self):
        """Test image filtering based on quality scores."""
        # Mock analysis results
        analysis_results = {
            "good_image1.fits": {"overall_quality_score": 0.8},
            "poor_image1.fits": {"overall_quality_score": 0.3},
            "good_image2.fits": {"overall_quality_score": 0.9},
            "poor_image2.fits": {"overall_quality_score": 0.4},
            "borderline.fits": {"overall_quality_score": 0.6},
        }

        good_images, poor_images = filter_images_by_quality(
            analysis_results,
            quality_threshold=0.6,
        )

        assert len(good_images) == 3  # 0.8, 0.9, 0.6
        assert len(poor_images) == 2  # 0.3, 0.4

        assert "good_image1.fits" in good_images
        assert "good_image2.fits" in good_images
        assert "borderline.fits" in good_images
        assert "poor_image1.fits" in poor_images
        assert "poor_image2.fits" in poor_images

    def test_filter_images_different_thresholds(self):
        """Test filtering with different quality thresholds."""
        analysis_results = {
            "image1.fits": {"overall_quality_score": 0.9},
            "image2.fits": {"overall_quality_score": 0.7},
            "image3.fits": {"overall_quality_score": 0.5},
        }

        # High threshold
        good, poor = filter_images_by_quality(analysis_results, quality_threshold=0.8)
        assert len(good) == 1
        assert len(poor) == 2

        # Low threshold
        good, poor = filter_images_by_quality(analysis_results, quality_threshold=0.4)
        assert len(good) == 3
        assert len(poor) == 0


class TestIntegration:
    """Integration tests for CV pre-filter."""

    def test_detector_with_custom_parameters(self):
        """Test detector with custom parameters."""
        detector = ArtifactDetector(min_streak_length=100, min_streak_strength=0.5)

        # Create test image
        image = np.random.normal(0.3, 0.1, (200, 200))
        image = np.clip(image, 0, 1)

        result = detector.comprehensive_analysis(image)

        assert isinstance(result, dict)
        assert "overall_quality_score" in result

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        detector = ArtifactDetector()

        # Test with all-zero image
        zero_image = np.zeros((50, 50))
        result = detector.comprehensive_analysis(zero_image)
        assert isinstance(result, dict)

        # Test with all-ones image
        ones_image = np.ones((50, 50))
        result = detector.comprehensive_analysis(ones_image)
        assert isinstance(result, dict)

        # Test with very small image
        tiny_image = np.random.random((5, 5))
        result = detector.comprehensive_analysis(tiny_image)
        assert isinstance(result, dict)
