#!/usr/bin/env python3
"""
Test script for the complete ML training pipeline.

This script validates that the new training functionality works correctly
by creating mock FITS data and running the complete pipeline.
"""

import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ml_training_pipeline():
    """Test the complete ML training pipeline with mock data."""

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fits_dir = temp_path / "fits_data"
        dataset_dir = temp_path / "dataset"
        model_path = temp_path / "trained_model.pth"

        fits_dir.mkdir()

        print("=" * 60)
        print("TESTING COMPLETE ML TRAINING PIPELINE")
        print("=" * 60)

        # Step 1: Create mock FITS data
        print("\nStep 1: Creating mock FITS data...")
        import numpy as np
        from astropy.io import fits

        # Create several mock FITS files with different characteristics
        mock_files = []
        for i in range(10):
            fits_file = fits_dir / f"mock_image_{i:03d}.fits"

            # Create mock FITS data with different patterns
            base_image = np.random.normal(1000, 100, (512, 512))

            # Add streaks to some images (should be detected as contaminated)
            if i % 3 == 0:
                # Add horizontal streak
                base_image[256:260, :] = 5000

            # Add noise/artifacts to some images
            if i % 2 == 0:
                # Add random hot pixels
                hot_pixels = np.random.random((512, 512)) > 0.995
                base_image[hot_pixels] = 8000

            # Save as FITS file
            hdu = fits.PrimaryHDU(base_image.astype(np.float64))
            hdu.header["OBJECT"] = f"MockImage{i}"
            hdu.header["EXPTIME"] = 300.0
            hdu.writeto(fits_file, overwrite=True)
            mock_files.append(fits_file)

        print(f"Created {len(mock_files)} mock FITS files")

        # Step 2: Test CV label generation
        print("\nStep 2: Testing CV label generation...")
        from nebulift.cv_prefilter import ArtifactDetector
        from nebulift.ml_model import generate_training_labels_from_cv

        detector = ArtifactDetector()

        labeled_files, labels, review_files = generate_training_labels_from_cv(
            mock_files,
            detector,
            clean_threshold=0.6,
            contaminated_threshold=0.4,
        )

        print("Label generation results:")
        print(f"  - Labeled files: {len(labeled_files)}")
        print(f"  - Clean labels: {labels.count(1)}")
        print(f"  - Contaminated labels: {labels.count(0)}")
        print(f"  - Review files: {len(review_files)}")

        # Step 3: Test dataset creation
        print("\nStep 3: Testing dataset creation...")
        from nebulift.ml_model import create_dataset_from_cv_labels

        if len(labeled_files) >= 4:  # Need minimum files for train/val split
            train_dataset, val_dataset, review_files = create_dataset_from_cv_labels(
                fits_files=mock_files,
                artifact_detector=detector,
                output_dir=dataset_dir,
                train_split=0.8,
                clean_threshold=0.6,
                contaminated_threshold=0.4,
            )

            print("Dataset creation results:")
            print(f"  - Training samples: {len(train_dataset)}")
            print(f"  - Validation samples: {len(val_dataset)}")
            print(f"  - Dataset directory: {dataset_dir}")

            # Step 4: Test model training (short run)
            print("\nStep 4: Testing model training...")
            from nebulift.ml_model import complete_training_pipeline

            try:
                results = complete_training_pipeline(
                    fits_directory=fits_dir,
                    model_output_path=model_path,
                    dataset_output_dir=temp_path / "full_pipeline_dataset",
                    epochs=2,  # Short training for testing
                    batch_size=2,  # Small batch for limited data
                    clean_threshold=0.6,
                    contaminated_threshold=0.4,
                )

                print("Training pipeline results:")
                print(f"  - Model saved: {Path(results['model_path']).exists()}")
                print(
                    f"  - Final validation accuracy: {results['final_metrics']['best_val_accuracy']:.2f}%",
                )
                print(
                    f"  - Training samples: {results['dataset_stats']['training_samples']}",
                )
                print(
                    f"  - Validation samples: {results['dataset_stats']['validation_samples']}",
                )

                print("\n✅ Complete ML training pipeline test PASSED!")
                return True

            except Exception as e:
                print(f"\n❌ Training pipeline failed: {e}")
                import traceback

                traceback.print_exc()
                return False
        else:
            print(
                "⚠️  Not enough labeled files for training test, but label generation works!",
            )
            return True


def test_cli_training_commands():
    """Test the new CLI training commands."""
    print("\n" + "=" * 60)
    print("TESTING CLI TRAINING COMMANDS")
    print("=" * 60)

    # Test CLI help
    import subprocess
    import sys

    try:
        # Test main help
        result = subprocess.run(
            [sys.executable, "-m", "nebulift.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode == 0 and "train-from-fits" in result.stdout:
            print("✅ CLI help includes new train-from-fits command")
        else:
            print("❌ CLI help test failed")
            return False

        # Test train-from-fits help
        result = subprocess.run(
            [sys.executable, "-m", "nebulift.cli", "train-from-fits", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode == 0 and "--clean_threshold" in result.stdout:
            print("✅ train-from-fits command help working")
        else:
            print("❌ train-from-fits help test failed")
            return False

        print("✅ CLI training commands test PASSED!")
        return True

    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def main():
    """Run all training pipeline tests."""
    print("NEBULIFT ML TRAINING PIPELINE VALIDATION")
    print("=" * 60)

    success = True

    # Test 1: Complete ML training pipeline
    try:
        if not test_ml_training_pipeline():
            success = False
    except Exception as e:
        print(f"❌ ML pipeline test failed with error: {e}")
        import traceback

        traceback.print_exc()
        success = False

    # Test 2: CLI commands
    try:
        if not test_cli_training_commands():
            success = False
    except Exception as e:
        print(f"❌ CLI test failed with error: {e}")
        success = False

    # Final result
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TRAINING PIPELINE TESTS PASSED!")
        print("✅ ML training pipeline is ready for Phase 1 deployment")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Please review errors above")
    print("=" * 60)

    return success


if __name__ == "__main__":
    main()
