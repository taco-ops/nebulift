#!/usr/bin/env python3
"""
Nebulift CLI - Command-line interface for astrophotography quality assessment

Usage:
    nebulift analyze <fits_file>                    # Analyze single FITS file
    nebulift batch <input_dir> <output_dir>         # Batch process directory
    nebulift train <data_dir> --model_output <path> # Train model on organized data
    nebulift train-from-fits <fits_dir> --model_output <path> --dataset_dir <path>  # Complete training pipeline
    nebulift k8s-train <config>                     # Launch distributed training
    nebulift validate                               # Run system validation
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from nebulift.cv_prefilter import ArtifactDetector
from nebulift.fits_processor import FITSProcessor

from .cv_prefilter import batch_analyze_images
from .ml_model import AstroQualityClassifier


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def analyze_single_file(fits_path: Path) -> dict:
    """Analyze a single FITS file for quality."""
    processor = FITSProcessor()
    detector = ArtifactDetector()

    # Load and process FITS file
    fits_data = processor.load_fits_file(fits_path)
    if not fits_data:
        return {"error": f"Failed to load FITS file: {fits_path}"}

    normalized = processor.normalize_image(fits_data["image_data"])
    analysis = detector.comprehensive_analysis(normalized)

    return {
        "file": str(fits_path),
        "quality_score": analysis["overall_quality_score"],
        "has_streaks": analysis["streaks"]["has_streaks"],
        "has_clouds": analysis["clouds"]["has_clouds"],
        "has_saturation": analysis["saturation"]["has_saturation"],
        "needs_review": analysis["needs_manual_review"],
        "recommendation": (
            "keep" if analysis["overall_quality_score"] > 0.5 else "discard"
        ),
    }


def batch_process(input_dir: Path, output_dir: Path) -> None:
    """Batch process FITS files in a directory."""
    detector = ArtifactDetector()

    # Create output directories
    clean_dir = output_dir / "clean"
    contaminated_dir = output_dir / "contaminated"
    review_dir = output_dir / "review"

    for dir_path in [clean_dir, contaminated_dir, review_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Find all FITS files
    fits_files = list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fit"))

    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return

    print(f"Processing {len(fits_files)} FITS files...")

    # Batch analyze
    results = batch_analyze_images([str(f) for f in fits_files], detector)

    # Sort files by quality
    for fits_path_str, analysis in results.items():
        fits_path = Path(fits_path_str)
        score = analysis["overall_quality_score"]

        if score > 0.7:
            # High quality - copy to clean directory
            clean_dir = output_dir / "clean"
            clean_dir.mkdir(exist_ok=True)
            shutil.copy2(fits_path, clean_dir / fits_path.name)
        elif score < 0.3:
            # Low quality - copy to contaminated directory
            contaminated_dir = output_dir / "contaminated"
            contaminated_dir.mkdir(exist_ok=True)
            shutil.copy2(fits_path, contaminated_dir / fits_path.name)
        else:
            # Uncertain - copy to review directory
            review_dir = output_dir / "review"
            review_dir.mkdir(exist_ok=True)
            shutil.copy2(fits_path, review_dir / fits_path.name)

    print(f"\nProcessing complete! Results saved to {output_dir}")


def train_model(data_dir: Path, model_output: Path, epochs: int = 50) -> None:
    """Train the ML model locally."""
    print(f"Training model on data from {data_dir}")

    # Create model and trainer
    model = AstroQualityClassifier(num_classes=2, pretrained=True)
    # trainer = ModelTrainer(model)  # TODO: Implement full training pipeline

    # This is a simplified placeholder - would need proper dataset creation
    print("Note: This CLI function needs proper dataset implementation")
    print("Use the Jupyter notebooks for complete training workflow")


def train_from_fits(
    fits_dir: Path,
    model_output: Path,
    dataset_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
) -> None:
    """Train model using CV-generated labels from FITS files."""
    print(f"Starting complete training pipeline from {fits_dir}")
    print(
        f"Clean threshold: {clean_threshold}, Contaminated threshold: {contaminated_threshold}",
    )

    print("Note: This CLI function needs proper pipeline implementation")
    print("Use the Jupyter notebooks for complete training workflow")


def launch_k8s_training(config_path: Optional[Path] = None) -> None:
    """Launch distributed training on Kubernetes."""
    if config_path and config_path.exists():
        print(f"Using config from {config_path}")
        # TODO: Parse custom config

    print("Launching Kubernetes distributed training...")
    print("Run these commands:")
    print()
    print("# Build Docker image")
    print("docker build -t nebulift:latest .")
    print()
    print("# Deploy to Kubernetes")
    print("kubectl apply -f k8s/")
    print()
    print("# Monitor training")
    print("kubectl logs -f job/nebulift-training")
    print()
    print("# Check status")
    print("kubectl get pods -l app=nebulift-training")


def validate_system() -> None:
    """Run system validation tests."""
    print("Running Nebulift system validation...")

    try:
        # Run validation
        result = subprocess.run(
            [sys.executable, "validate_system.py"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print("✅ System validation passed!")
            print(result.stdout)
        else:
            print("❌ System validation failed!")
            print(result.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error running validation: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Nebulift: Astrophotography Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze single FITS file")
    analyze_parser.add_argument("fits_file", type=Path, help="Path to FITS file")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process directory")
    batch_parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with FITS files",
    )
    batch_parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for sorted files",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train model locally on organized data",
    )
    train_parser.add_argument(
        "data_dir",
        type=Path,
        help="Training data directory (with train/val subdirs)",
    )
    train_parser.add_argument(
        "--model_output",
        type=Path,
        required=True,
        help="Output path for trained model",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    # Train from FITS command (new complete pipeline)
    fits_train_parser = subparsers.add_parser(
        "train-from-fits",
        help="Complete training pipeline from FITS files using CV labels",
    )
    fits_train_parser.add_argument(
        "fits_dir",
        type=Path,
        help="Directory containing FITS files",
    )
    fits_train_parser.add_argument(
        "--model_output",
        type=Path,
        required=True,
        help="Output path for trained model",
    )
    fits_train_parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Directory to save organized dataset",
    )
    fits_train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    fits_train_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    fits_train_parser.add_argument(
        "--clean_threshold",
        type=float,
        default=0.7,
        help="Quality threshold for clean label (default: 0.7)",
    )
    fits_train_parser.add_argument(
        "--contaminated_threshold",
        type=float,
        default=0.3,
        help="Quality threshold for contaminated label (default: 0.3)",
    )

    # K8s train command
    k8s_parser = subparsers.add_parser("k8s-train", help="Launch distributed training")
    k8s_parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Optional config file path",
    )

    # Validate command
    subparsers.add_parser("validate", help="Run system validation")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    try:
        if args.command == "analyze":
            result = analyze_single_file(args.fits_file)
            if "error" in result:
                print(f"Error: {result['error']}")
                sys.exit(1)
            else:
                print(f"File: {result['file']}")
                print(f"Quality Score: {result['quality_score']:.3f}")
                print(f"Has Streaks: {result['has_streaks']}")
                print(f"Has Clouds: {result['has_clouds']}")
                print(f"Has Saturation: {result['has_saturation']}")
                print(f"Needs Review: {result['needs_review']}")
                print(f"Recommendation: {result['recommendation']}")

        elif args.command == "batch":
            batch_process(args.input_dir, args.output_dir)

        elif args.command == "train":
            train_model(args.data_dir, args.model_output, args.epochs)

        elif args.command == "train-from-fits":
            train_from_fits(
                args.fits_dir,
                args.model_output,
                args.dataset_dir,
                args.epochs,
                args.batch_size,
                args.clean_threshold,
                args.contaminated_threshold,
            )

        elif args.command == "k8s-train":
            launch_k8s_training(args.config)

        elif args.command == "validate":
            validate_system()

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
