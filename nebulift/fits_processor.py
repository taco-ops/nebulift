"""
FITS Image Processor Module

Handles loading, processing, and normalization of astronomical FITS files
for machine learning quality assessment.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

logger = logging.getLogger(__name__)


class FITSProcessor:
    """Processes FITS astronomical images for quality assessment."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize FITS processor.
        
        Args:
            target_size: Target image size for ML processing (width, height)
        """
        self.target_size = target_size
        self.stats_cache: dict[str, dict[str, float]] = {}

    def load_fits_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a FITS file and extract image data with metadata.
        
        Args:
            file_path: Path to FITS file
            
        Returns:
            Dictionary containing image data and metadata, or None if failed
        """
        try:
            with fits.open(file_path) as hdul:
                # Get primary HDU (Header Data Unit)
                primary_hdu = hdul[0]
                image_data = primary_hdu.data
                header = primary_hdu.header

                if image_data is None:
                    logger.warning(f"No image data found in {file_path}")
                    return None

                # Convert to float64 for processing
                image_data = image_data.astype(np.float64)

                # Extract relevant metadata
                metadata = self._extract_metadata(header)

                return {
                    "image_data": image_data,
                    "header": header,
                    "metadata": metadata,
                    "file_path": file_path,
                }

        except Exception as e:
            logger.error(f"Failed to load FITS file {file_path}: {e}")
            return None

    def _extract_metadata(self, header: fits.Header) -> Dict[str, Any]:
        """Extract relevant metadata from FITS header."""
        metadata = {}

        # Standard FITS keywords
        standard_keys = [
            "EXPTIME", "EXPOSURE",  # Exposure time
            "GAIN",                 # Camera gain
            "TEMP", "CCD-TEMP",    # CCD temperature
            "FILTER",              # Filter used
            "OBJECT",              # Target object
            "DATE-OBS",            # Observation date
            "TELESCOP",            # Telescope
            "INSTRUME",            # Instrument
        ]

        for key in standard_keys:
            if key in header:
                metadata[key.lower().replace("-", "_")] = header[key]

        # Image dimensions
        if "NAXIS1" in header and "NAXIS2" in header:
            metadata["width"] = header["NAXIS1"]
            metadata["height"] = header["NAXIS2"]

        return metadata

    def normalize_image(self, image_data: np.ndarray,
                       method: str = "percentile") -> np.ndarray:
        """
        Normalize image data for consistent processing.
        
        Args:
            image_data: Raw image data
            method: Normalization method ('percentile', 'sigma_clip', 'minmax')
            
        Returns:
            Normalized image data in range [0, 1]
        """
        if method == "percentile":
            # Use percentile normalization (robust to outliers)
            p1, p99 = np.percentile(image_data, [1, 99])
            if p99 > p1:
                normalized = np.clip((image_data - p1) / (p99 - p1), 0, 1)
            else:
                # Handle case where image has uniform values
                normalized = np.zeros_like(image_data, dtype=np.float64)

        elif method == "sigma_clip":
            # Use sigma clipping to remove outliers
            clipped = sigma_clip(image_data, sigma=3, maxiters=5)
            vmin, vmax = clipped.min(), clipped.max()
            normalized = np.clip((image_data - vmin) / (vmax - vmin), 0, 1)

        elif method == "minmax":
            # Simple min-max normalization
            vmin, vmax = image_data.min(), image_data.max()
            if vmax > vmin:
                normalized = (image_data - vmin) / (vmax - vmin)
            else:
                normalized = np.zeros_like(image_data)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def resize_for_ml(self, image_data: np.ndarray) -> np.ndarray:
        """
        Resize image for machine learning processing.
        
        Args:
            image_data: Normalized image data
            
        Returns:
            Resized image as numpy array
        """
        # Convert to uint8 for OpenCV
        image_uint8 = (image_data * 255).astype(np.uint8)

        # Resize using OpenCV with anti-aliasing
        resized = cv2.resize(image_uint8, self.target_size,
                           interpolation=cv2.INTER_AREA)

        # Convert back to float32 for ML
        return resized.astype(np.float32) / 255.0

    def compute_image_stats(self, image_data: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical properties of the image.
        
        Args:
            image_data: Image data array
            
        Returns:
            Dictionary of image statistics
        """
        # Use sigma clipping for robust statistics
        clipped = sigma_clip(image_data, sigma=3, maxiters=5)

        stats = {
            "mean": float(np.mean(clipped)),
            "std": float(np.std(clipped)),
            "median": float(np.median(clipped)),
            "min": float(np.min(image_data)),
            "max": float(np.max(image_data)),
            "p01": float(np.percentile(image_data, 1)),
            "p99": float(np.percentile(image_data, 99)),
            "dynamic_range": float(np.percentile(image_data, 99) -
                                 np.percentile(image_data, 1)),
        }

        return stats

    def process_fits_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Complete processing pipeline for a FITS file.
        
        Args:
            file_path: Path to FITS file
            
        Returns:
            Processed image data and metadata
        """
        # Load FITS file
        fits_data = self.load_fits_file(file_path)
        if fits_data is None:
            return None

        image_data = fits_data["image_data"]

        # Normalize image
        normalized = self.normalize_image(image_data)

        # Resize for ML
        ml_ready = self.resize_for_ml(normalized)

        # Compute statistics
        stats = self.compute_image_stats(image_data)

        return {
            "original_image": image_data,
            "normalized_image": normalized,
            "ml_image": ml_ready,
            "metadata": fits_data["metadata"],
            "stats": stats,
            "file_path": file_path,
        }


def validate_fits_file(file_path: Path) -> bool:
    """
    Quick validation of FITS file without full loading.
    
    Args:
        file_path: Path to FITS file
        
    Returns:
        True if file appears to be a valid FITS file
    """
    try:
        with fits.open(file_path) as hdul:
            # Check if we have at least one HDU with image data
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    return True
        return False
    except Exception:
        return False


def batch_validate_fits_files(directory: Path) -> Tuple[list, list]:
    """
    Validate all FITS files in a directory.
    
    Args:
        directory: Directory containing FITS files
        
    Returns:
        Tuple of (valid_files, invalid_files)
    """
    fits_patterns = ["*.fits", "*.fit", "*.fts"]
    valid_files = []
    invalid_files = []

    for pattern in fits_patterns:
        for file_path in directory.glob(pattern):
            if validate_fits_file(file_path):
                valid_files.append(file_path)
            else:
                invalid_files.append(file_path)

    return valid_files, invalid_files
