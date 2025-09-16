"""
Test suite for FITS Image Processor

Tests the FITS file loading, processing, and normalization functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nebulift.fits_processor import (
    FITSProcessor,
    batch_validate_fits_files,
    validate_fits_file,
)


class TestFITSProcessor:
    """Test cases for FITSProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FITSProcessor(target_size=(224, 224))

    def test_init(self):
        """Test FITSProcessor initialization."""
        processor = FITSProcessor(target_size=(512, 512))
        assert processor.target_size == (512, 512)
        assert isinstance(processor.stats_cache, dict)

    @patch("nebulift.fits_processor.fits")
    def test_load_fits_file_success(self, mock_fits):
        """Test successful FITS file loading."""
        # Mock FITS data
        mock_image_data = np.random.random((1000, 1000)).astype(np.float64)
        mock_header = {"EXPTIME": 30.0, "GAIN": 1.0, "NAXIS1": 1000, "NAXIS2": 1000}

        mock_hdu = Mock()
        mock_hdu.data = mock_image_data
        mock_hdu.header = mock_header

        mock_hdul = Mock()
        mock_hdul.__enter__ = Mock(return_value=[mock_hdu])
        mock_hdul.__exit__ = Mock(return_value=None)
        mock_fits.open.return_value = mock_hdul

        # Test loading
        file_path = Path("test.fits")
        result = self.processor.load_fits_file(file_path)

        assert result is not None
        assert "image_data" in result
        assert "metadata" in result
        assert "file_path" in result
        assert result["file_path"] == file_path
        np.testing.assert_array_equal(result["image_data"], mock_image_data)

    @patch("nebulift.fits_processor.fits")
    def test_load_fits_file_no_data(self, mock_fits):
        """Test FITS file with no image data."""
        mock_hdu = Mock()
        mock_hdu.data = None
        mock_hdu.header = {}

        mock_hdul = Mock()
        mock_hdul.__enter__ = Mock(return_value=[mock_hdu])
        mock_hdul.__exit__ = Mock(return_value=None)
        mock_fits.open.return_value = mock_hdul

        result = self.processor.load_fits_file(Path("test.fits"))
        assert result is None

    @patch("nebulift.fits_processor.fits")
    def test_load_fits_file_exception(self, mock_fits):
        """Test FITS file loading with exception."""
        mock_fits.open.side_effect = Exception("File not found")

        result = self.processor.load_fits_file(Path("nonexistent.fits"))
        assert result is None

    def test_extract_metadata(self):
        """Test metadata extraction from FITS header."""
        mock_header = {
            "EXPTIME": 30.0,
            "GAIN": 1.5,
            "CCD-TEMP": -10.0,
            "FILTER": "V",
            "NAXIS1": 1000,
            "NAXIS2": 1000,
            "UNKNOWN": "should_not_appear",
        }

        metadata = self.processor._extract_metadata(mock_header)

        assert metadata["exptime"] == 30.0
        assert metadata["gain"] == 1.5
        assert metadata["ccd_temp"] == -10.0
        assert metadata["filter"] == "V"
        assert metadata["width"] == 1000
        assert metadata["height"] == 1000
        assert "unknown" not in metadata

    @given(
        width=st.integers(min_value=10, max_value=100),
        height=st.integers(min_value=10, max_value=100),
        min_val=st.floats(min_value=0, max_value=1000),
        max_val=st.floats(min_value=1001, max_value=65535),
    )
    def test_normalize_image_percentile(self, width, height, min_val, max_val):
        """Property-based test for percentile normalization."""
        # Create proper random image data
        image_data = np.random.uniform(min_val, max_val, (height, width))

        normalized = self.processor.normalize_image(image_data, method="percentile")

        # Check output range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.shape == image_data.shape

    def test_normalize_image_methods(self):
        """Test different normalization methods."""
        # Create test image with known statistics
        image_data = np.random.normal(1000, 200, (100, 100))
        image_data[50:55, 50:55] = 10000  # Add some outliers

        # Test percentile normalization
        norm_percentile = self.processor.normalize_image(
            image_data,
            method="percentile",
        )
        assert 0 <= norm_percentile.min() <= norm_percentile.max() <= 1

        # Test sigma clip normalization
        norm_sigma = self.processor.normalize_image(image_data, method="sigma_clip")
        assert 0 <= norm_sigma.min() <= norm_sigma.max() <= 1

        # Test minmax normalization
        norm_minmax = self.processor.normalize_image(image_data, method="minmax")
        assert 0 <= norm_minmax.min() <= norm_minmax.max() <= 1

        # Test invalid method
        with pytest.raises(ValueError):
            self.processor.normalize_image(image_data, method="invalid")

    def test_resize_for_ml(self):
        """Test image resizing for ML processing."""
        # Create test image
        image_data = np.random.random((1000, 1500))

        resized = self.processor.resize_for_ml(image_data)

        assert resized.shape == (224, 224)
        assert resized.dtype == np.float32
        assert 0 <= resized.min() <= resized.max() <= 1

    def test_compute_image_stats(self):
        """Test image statistics computation."""
        # Create test image with known properties
        image_data = np.random.normal(1000, 100, (500, 500))

        stats = self.processor.compute_image_stats(image_data)

        required_keys = [
            "mean",
            "std",
            "median",
            "min",
            "max",
            "p01",
            "p99",
            "dynamic_range",
        ]
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], float)

        # Check logical relationships
        assert (
            stats["min"]
            <= stats["p01"]
            <= stats["median"]
            <= stats["p99"]
            <= stats["max"]
        )
        assert stats["dynamic_range"] == stats["p99"] - stats["p01"]

    @patch("nebulift.fits_processor.FITSProcessor.load_fits_file")
    def test_process_fits_file_success(self, mock_load):
        """Test complete FITS file processing pipeline."""
        # Mock loaded FITS data
        image_data = np.random.random((1000, 1000)) * 1000
        mock_load.return_value = {
            "image_data": image_data,
            "metadata": {"exptime": 30.0},
        }

        result = self.processor.process_fits_file(Path("test.fits"))

        assert result is not None
        assert "original_image" in result
        assert "normalized_image" in result
        assert "ml_image" in result
        assert "metadata" in result
        assert "stats" in result
        assert "file_path" in result

        # Check ML image shape
        assert result["ml_image"].shape == (224, 224)

    @patch("nebulift.fits_processor.FITSProcessor.load_fits_file")
    def test_process_fits_file_failure(self, mock_load):
        """Test processing with failed file loading."""
        mock_load.return_value = None

        result = self.processor.process_fits_file(Path("bad.fits"))
        assert result is None


class TestValidationFunctions:
    """Test cases for FITS file validation functions."""

    @patch("nebulift.fits_processor.fits")
    def test_validate_fits_file_valid(self, mock_fits):
        """Test validation of valid FITS file."""
        mock_hdu = Mock()
        mock_hdu.data = np.random.random((1000, 1000))

        mock_hdul = Mock()
        mock_hdul.__enter__ = Mock(return_value=[mock_hdu])
        mock_hdul.__exit__ = Mock(return_value=None)
        mock_hdul.__iter__ = Mock(return_value=iter([mock_hdu]))
        mock_fits.open.return_value = mock_hdul

        assert validate_fits_file(Path("test.fits")) is True

    @patch("nebulift.fits_processor.fits")
    def test_validate_fits_file_invalid(self, mock_fits):
        """Test validation of invalid FITS file."""
        mock_fits.open.side_effect = Exception("Invalid file")

        assert validate_fits_file(Path("bad.fits")) is False

    @patch("nebulift.fits_processor.validate_fits_file")
    def test_batch_validate_fits_files(self, mock_validate):
        """Test batch validation of FITS files."""
        # Mock validation results
        mock_validate.side_effect = lambda x: str(x).endswith("good.fits")

        # Mock directory with files
        mock_directory = Mock()
        mock_files = [
            Path("image1_good.fits"),
            Path("image2_bad.fits"),
            Path("image3_good.fits"),
        ]

        # Mock glob for each pattern separately to avoid duplication
        def mock_glob(pattern):
            if pattern == "*.fits":
                return mock_files
            return []  # No files for other patterns

        mock_directory.glob.side_effect = mock_glob

        valid_files, invalid_files = batch_validate_fits_files(mock_directory)

        assert len(valid_files) == 2
        assert len(invalid_files) == 1
        assert Path("image1_good.fits") in valid_files
        assert Path("image3_good.fits") in valid_files
        assert Path("image2_bad.fits") in invalid_files


class TestIntegration:
    """Integration tests for FITS processing."""

    def test_create_mock_fits_data(self):
        """Test helper for creating mock FITS data."""
        # This would create actual FITS files for integration testing
        # For now, just ensure the test framework works
        processor = FITSProcessor()
        assert processor.target_size == (224, 224)

    @pytest.mark.slow
    def test_memory_usage_large_batch(self):
        """Test memory usage with large batch of images."""
        # This test would process many images to check memory efficiency
        # Marked as slow test to run separately
