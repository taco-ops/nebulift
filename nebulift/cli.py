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
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from nebulift.cv_prefilter import ArtifactDetector
from nebulift.fits_processor import FITSProcessor

from .cv_prefilter import batch_analyze_images
from .ml_model import (
    AstroQualityClassifier,
    QualityPredictor,
    complete_training_pipeline,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def classify_cv_score(
    quality_score: float,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
) -> str:
    """Classify a CV quality score into a sorting bucket."""
    if contaminated_threshold >= clean_threshold:
        raise ValueError("contaminated_threshold must be lower than clean_threshold")
    if quality_score >= clean_threshold:
        return "clean"
    if quality_score <= contaminated_threshold:
        return "contaminated"
    return "review"


def analyze_single_file(
    fits_path: Path,
    model_path: Optional[Path] = None,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
) -> dict[str, Any]:
    """Analyze a single FITS file for quality."""
    processor = FITSProcessor()
    detector = ArtifactDetector()

    # Load and process FITS file
    fits_data = processor.load_fits_file(fits_path)
    if not fits_data:
        return {"error": f"Failed to load FITS file: {fits_path}"}

    normalized = processor.normalize_image(fits_data["image_data"])
    analysis = detector.comprehensive_analysis(normalized)
    quality_score = float(analysis["overall_quality_score"])
    cv_label = classify_cv_score(
        quality_score,
        clean_threshold=clean_threshold,
        contaminated_threshold=contaminated_threshold,
    )

    ml_prediction = None
    decision_label = cv_label
    decision_source = "cv"
    if model_path is not None:
        predictor = QualityPredictor(str(model_path))
        ml_prediction = predictor.predict_single(str(fits_path), processor)
        if "error" not in ml_prediction:
            decision_label = str(ml_prediction["predicted_label"])
            decision_source = "ml"

    return {
        "file": str(fits_path),
        "quality_score": quality_score,
        "has_streaks": analysis["streaks"]["has_streaks"],
        "has_clouds": analysis["clouds"]["has_clouds"],
        "has_saturation": analysis["saturation"]["has_saturation"],
        "needs_review": analysis["needs_manual_review"],
        "cv_label": cv_label,
        "decision_label": decision_label,
        "decision_source": decision_source,
        "ml_prediction": ml_prediction,
    }


def _manifest_summary(entries: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"clean": 0, "contaminated": 0, "review": 0, "errors": 0}
    for entry in entries:
        if entry.get("error"):
            summary["errors"] += 1
        else:
            label = str(entry["decision_label"])
            summary[label] = summary.get(label, 0) + 1
    return summary


def _build_manifest_entry(
    fits_path: Path,
    analysis: dict[str, Any],
    cv_label: str,
    decision_label: str,
    decision_source: str,
    action: str,
    destination_path: Optional[Path],
    ml_prediction: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    quality_score = float(analysis["overall_quality_score"])
    return {
        "source_path": str(fits_path),
        "file_name": fits_path.name,
        "quality_score": quality_score,
        "cv_label": cv_label,
        "decision_label": decision_label,
        "decision_source": decision_source,
        "action": action,
        "destination_path": str(destination_path) if destination_path else None,
        "needs_review": bool(analysis["needs_manual_review"]),
        "artifacts": {
            "has_streaks": bool(analysis["streaks"]["has_streaks"]),
            "has_clouds": bool(analysis["clouds"]["has_clouds"]),
            "has_saturation": bool(analysis["saturation"]["has_saturation"]),
            "has_hot_pixels": bool(analysis["hot_pixels"]["has_hot_pixels"]),
        },
        "ml_prediction": ml_prediction,
        "reviewed": False,
        "corrected_label": None,
    }


def batch_process(
    input_dir: Path,
    output_dir: Path,
    action: str = "report",
    model_path: Optional[Path] = None,
    manifest_name: str = "batch_manifest.json",
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
) -> Path:
    """Batch process FITS files and write a JSON manifest."""
    if action not in {"report", "move"}:
        raise ValueError("action must be 'report' or 'move'")

    detector = ArtifactDetector()
    processor = FITSProcessor()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all FITS files
    fits_files = sorted(
        list(input_dir.glob("*.fits"))
        + list(input_dir.glob("*.fit"))
        + list(input_dir.glob("*.fts")),
    )

    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        manifest_path = output_dir / manifest_name
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "action": action,
                    "model_path": str(model_path) if model_path else None,
                    "summary": {
                        "clean": 0,
                        "contaminated": 0,
                        "review": 0,
                        "errors": 0,
                    },
                    "files": [],
                },
                indent=2,
            ),
        )
        return manifest_path

    print(f"Processing {len(fits_files)} FITS files...")

    # Batch analyze
    results = batch_analyze_images([str(f) for f in fits_files], detector, processor)
    predictor = QualityPredictor(str(model_path)) if model_path else None
    entries = []

    for fits_path in fits_files:
        analysis = results.get(str(fits_path))
        if analysis is None:
            entries.append(
                {
                    "source_path": str(fits_path),
                    "file_name": fits_path.name,
                    "error": "analysis_failed",
                    "action": action,
                    "destination_path": None,
                    "reviewed": False,
                    "corrected_label": None,
                },
            )
            continue

        quality_score = float(analysis["overall_quality_score"])
        decision_label = classify_cv_score(
            quality_score,
            clean_threshold=clean_threshold,
            contaminated_threshold=contaminated_threshold,
        )
        decision_source = "cv"
        ml_prediction = None

        if predictor is not None:
            ml_prediction = predictor.predict_single(str(fits_path), processor)
            if "error" not in ml_prediction:
                decision_label = str(ml_prediction["predicted_label"])
                decision_source = "ml"

        destination_path = None
        if action == "move":
            destination_dir = output_dir / decision_label
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / fits_path.name
            shutil.move(str(fits_path), str(destination_path))

        entries.append(
            _build_manifest_entry(
                fits_path=fits_path,
                analysis=analysis,
                cv_label=classify_cv_score(
                    quality_score,
                    clean_threshold=clean_threshold,
                    contaminated_threshold=contaminated_threshold,
                ),
                decision_label=decision_label,
                decision_source=decision_source,
                action=action,
                destination_path=destination_path,
                ml_prediction=ml_prediction,
            ),
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "action": action,
        "model_path": str(model_path) if model_path else None,
        "thresholds": {
            "clean": clean_threshold,
            "contaminated": contaminated_threshold,
        },
        "summary": _manifest_summary(entries),
        "files": entries,
    }
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nProcessing complete. Manifest saved to {manifest_path}")
    if action == "move":
        print(f"Files moved into buckets under {output_dir}")
    return manifest_path


def review_manifest(manifest_path: Path) -> None:
    """Interactively review and correct labels in a batch manifest."""
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", [])
    label_choices = {
        "c": "clean",
        "clean": "clean",
        "x": "contaminated",
        "contaminated": "contaminated",
        "r": "review",
        "review": "review",
    }

    for index, entry in enumerate(files, start=1):
        if entry.get("error"):
            continue

        current_label = entry.get("corrected_label") or entry.get("decision_label")
        print()
        print(f"[{index}/{len(files)}] {entry['source_path']}")
        print(f"Quality score: {entry.get('quality_score')}")
        print(
            f"Decision: {entry.get('decision_label')} ({entry.get('decision_source')})"
        )
        print(f"Current label: {current_label}")
        print(f"Artifacts: {entry.get('artifacts')}")
        response = input(
            "Set label [c=clean, x=contaminated, r=review, s=skip, q=quit]: "
        )
        response = response.strip().lower()

        if response == "q":
            break
        if response in {"", "s", "skip"}:
            continue
        if response not in label_choices:
            print(f"Unknown label choice: {response}")
            continue

        entry["corrected_label"] = label_choices[response]
        entry["reviewed"] = True

    manifest["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Updated manifest: {manifest_path}")


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
    results = complete_training_pipeline(
        fits_directory=fits_dir,
        model_output_path=model_output,
        dataset_output_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        clean_threshold=clean_threshold,
        contaminated_threshold=contaminated_threshold,
    )

    stats = results["dataset_stats"]
    metrics = results["final_metrics"]
    print("\nTraining complete!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Dataset manifests saved to: {results['dataset_dir']}")
    print(f"Training samples: {stats['training_samples']}")
    print(f"Validation samples: {stats['validation_samples']}")
    print(f"Review samples: {stats['review_samples']}")
    print(f"Best validation accuracy: {metrics['best_val_accuracy']:.2f}%")


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
    analyze_parser.add_argument(
        "--model",
        type=Path,
        help="Optional trained model checkpoint for ML-assisted classification",
    )
    analyze_parser.add_argument(
        "--clean_threshold",
        type=float,
        default=0.7,
        help="CV quality threshold for clean label (default: 0.7)",
    )
    analyze_parser.add_argument(
        "--contaminated_threshold",
        type=float,
        default=0.3,
        help="CV quality threshold for contaminated label (default: 0.3)",
    )

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
        help="Output directory for JSON manifest and optional moved files",
    )
    batch_parser.add_argument(
        "--action",
        choices=["report", "move"],
        default="report",
        help="Batch action: write report only or move files into buckets (default: report)",
    )
    batch_parser.add_argument(
        "--model",
        type=Path,
        help="Optional trained model checkpoint for ML-assisted classification",
    )
    batch_parser.add_argument(
        "--manifest",
        default="batch_manifest.json",
        help="Manifest filename to write under output_dir",
    )
    batch_parser.add_argument(
        "--clean_threshold",
        type=float,
        default=0.7,
        help="CV quality threshold for clean label (default: 0.7)",
    )
    batch_parser.add_argument(
        "--contaminated_threshold",
        type=float,
        default=0.3,
        help="CV quality threshold for contaminated label (default: 0.3)",
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

    # Review command
    review_parser = subparsers.add_parser(
        "review",
        help="Interactively review and correct a JSON batch manifest",
    )
    review_parser.add_argument(
        "manifest",
        type=Path,
        help="Path to a batch_manifest.json file",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    try:
        if args.command == "analyze":
            result = analyze_single_file(
                args.fits_file,
                model_path=args.model,
                clean_threshold=args.clean_threshold,
                contaminated_threshold=args.contaminated_threshold,
            )
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
                print(f"CV Label: {result['cv_label']}")
                print(f"Decision Label: {result['decision_label']}")
                print(f"Decision Source: {result['decision_source']}")

        elif args.command == "batch":
            batch_process(
                args.input_dir,
                args.output_dir,
                action=args.action,
                model_path=args.model,
                manifest_name=args.manifest,
                clean_threshold=args.clean_threshold,
                contaminated_threshold=args.contaminated_threshold,
            )

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

        elif args.command == "review":
            review_manifest(args.manifest)

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
