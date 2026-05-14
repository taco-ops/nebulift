#!/usr/bin/env python3
"""Nebulift CLI for astrophotography quality assessment."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import manifest as _manifest
from . import registry as _registry
from .manifest import (
    analyze_single_file,
    batch_process,
    calibrate_thresholds,
    evaluate_manifest,
    export_curated_dataset,
    print_calibration_report,
    print_evaluation_report,
    review_manifest,
)
from .registry import (
    DEFAULT_LOCAL_SETTINGS_PATH,
    DEFAULT_REGISTRY_PATH,
    print_model_registry,
    print_threshold_settings,
    promote_model,
    promote_thresholds,
    promote_thresholds_from_calibration,
    register_model,
)
from .training import train_from_fits, train_from_manifest, train_model

# Compatibility aliases for callers that imported helpers from nebulift.cli.
classify_cv_score = _manifest.classify_cv_score
open_file_in_viewer = _manifest.open_file_in_viewer
load_model_registry = _registry.load_model_registry
load_local_settings = _registry.load_local_settings
resolve_model_path = _registry.resolve_model_path
resolve_thresholds = _registry.resolve_thresholds


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def launch_k8s_training(config_path: Optional[Path] = None) -> None:
    """Print Kubernetes training launch guidance."""
    if config_path and config_path.exists():
        print(f"Using config from {config_path}")

    print("Launching Kubernetes distributed training requires a configured cluster.")
    print("Build image: docker build -t nebulift:latest .")
    print("Deploy: kubectl apply -f k8s/")
    print("Monitor: kubectl logs -f job/nebulift-training")
    print("Status: kubectl get pods -l app=nebulift-training")


def validate_system() -> None:
    """Run system validation tests."""
    print("Running Nebulift system validation...")
    try:
        result = subprocess.run(
            [sys.executable, "validate_system.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print("System validation passed.")
            print(result.stdout)
        else:
            print("System validation failed.")
            print(result.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error running validation: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Nebulift: Astrophotography Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze single FITS file")
    analyze_parser.add_argument("fits_file", type=Path, help="Path to FITS file")
    analyze_parser.add_argument("--model", type=Path, help="Optional model checkpoint")
    analyze_parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Model registry path for promoted defaults",
    )
    analyze_parser.add_argument(
        "--no-default-model",
        action="store_true",
        help="Disable use of a promoted default model",
    )
    analyze_parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_LOCAL_SETTINGS_PATH,
        help="Local settings path for promoted thresholds",
    )
    analyze_parser.add_argument(
        "--no-default-thresholds",
        action="store_true",
        help="Disable use of promoted default thresholds",
    )
    analyze_parser.add_argument("--clean_threshold", type=float)
    analyze_parser.add_argument("--contaminated_threshold", type=float)

    batch_parser = subparsers.add_parser("batch", help="Batch process directory")
    batch_parser.add_argument("input_dir", type=Path, help="Input FITS directory")
    batch_parser.add_argument("output_dir", type=Path, help="Output directory")
    batch_parser.add_argument(
        "--action",
        choices=["report", "move"],
        default="report",
        help="Write report only or move files into buckets",
    )
    batch_parser.add_argument("--model", type=Path, help="Optional model checkpoint")
    batch_parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Model registry path for promoted defaults",
    )
    batch_parser.add_argument(
        "--no-default-model",
        action="store_true",
        help="Disable use of a promoted default model",
    )
    batch_parser.add_argument("--manifest", default="batch_manifest.json")
    batch_parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_LOCAL_SETTINGS_PATH,
        help="Local settings path for promoted thresholds",
    )
    batch_parser.add_argument(
        "--no-default-thresholds",
        action="store_true",
        help="Disable use of promoted default thresholds",
    )
    batch_parser.add_argument("--clean_threshold", type=float)
    batch_parser.add_argument("--contaminated_threshold", type=float)

    train_parser = subparsers.add_parser(
        "train", help="Train from curated class folders"
    )
    train_parser.add_argument("data_dir", type=Path)
    train_parser.add_argument("--model_output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--train_split", type=float, default=0.8)

    fits_train_parser = subparsers.add_parser(
        "train-from-fits",
        help="Train from FITS files using CV labels",
    )
    fits_train_parser.add_argument("fits_dir", type=Path)
    fits_train_parser.add_argument("--model_output", type=Path, required=True)
    fits_train_parser.add_argument("--dataset_dir", type=Path, required=True)
    fits_train_parser.add_argument("--epochs", type=int, default=50)
    fits_train_parser.add_argument("--batch_size", type=int, default=32)
    fits_train_parser.add_argument("--clean_threshold", type=float, default=0.7)
    fits_train_parser.add_argument("--contaminated_threshold", type=float, default=0.3)

    manifest_train_parser = subparsers.add_parser(
        "train-from-manifest",
        help="Train from labels in a reviewed JSON batch manifest",
    )
    manifest_train_parser.add_argument("manifest", type=Path)
    manifest_train_parser.add_argument("--model_output", type=Path, required=True)
    manifest_train_parser.add_argument("--dataset_dir", type=Path, required=True)
    manifest_train_parser.add_argument("--epochs", type=int, default=50)
    manifest_train_parser.add_argument("--batch_size", type=int, default=32)
    manifest_train_parser.add_argument("--reviewed_only", action="store_true")

    subparsers.add_parser("validate", help="Run system validation")
    k8s_parser = subparsers.add_parser("k8s-train", help="Launch distributed training")
    k8s_parser.add_argument("config", type=Path, nargs="?")

    review_parser = subparsers.add_parser("review", help="Review a batch manifest")
    review_parser.add_argument("manifest", type=Path)
    review_parser.add_argument("--open", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate-manifest",
        help="Evaluate manifest decisions against corrected labels",
    )
    evaluate_parser.add_argument("manifest", type=Path)
    evaluate_parser.add_argument(
        "--prediction",
        choices=["decision", "cv"],
        default="decision",
    )
    evaluate_parser.add_argument("--output", type=Path)

    calibrate_parser = subparsers.add_parser(
        "calibrate-thresholds",
        help="Calibrate CV thresholds from reviewed manifests",
    )
    calibrate_parser.add_argument("manifests", type=Path, nargs="+")
    calibrate_parser.add_argument("--step", type=float, default=0.05)
    calibrate_parser.add_argument("--min-gap", type=float, default=0.1)
    calibrate_parser.add_argument("--output", type=Path)

    threshold_settings_parser = subparsers.add_parser(
        "show-thresholds",
        help="Show promoted default CV thresholds",
    )
    threshold_settings_parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_LOCAL_SETTINGS_PATH,
    )

    promote_thresholds_parser = subparsers.add_parser(
        "promote-thresholds",
        help="Promote default CV thresholds",
    )
    promote_thresholds_parser.add_argument("--clean_threshold", type=float)
    promote_thresholds_parser.add_argument("--contaminated_threshold", type=float)
    promote_thresholds_parser.add_argument("--calibration", type=Path)
    promote_thresholds_parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_LOCAL_SETTINGS_PATH,
    )

    export_parser = subparsers.add_parser(
        "export-curated",
        help="Export reviewed manifest labels into class folders",
    )
    export_parser.add_argument("manifest", type=Path)
    export_parser.add_argument("output_dir", type=Path)
    export_parser.add_argument(
        "--action",
        choices=["copy", "symlink", "move"],
        default="copy",
    )
    export_parser.add_argument("--include-unreviewed", action="store_true")
    export_parser.add_argument(
        "--output-manifest",
        dest="manifest_name",
        default="curated_manifest.json",
    )

    register_model_parser = subparsers.add_parser(
        "register-model", help="Register model"
    )
    register_model_parser.add_argument("model_path", type=Path)
    register_model_parser.add_argument("--name", required=True)
    register_model_parser.add_argument("--model-id")
    register_model_parser.add_argument("--evaluation", type=Path)
    register_model_parser.add_argument("--calibration", type=Path)
    register_model_parser.add_argument("--source-manifest", type=Path)
    register_model_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH
    )
    register_model_parser.add_argument("--promote", action="store_true")
    register_model_parser.add_argument("--replace", action="store_true")

    promote_model_parser = subparsers.add_parser("promote-model", help="Promote model")
    promote_model_parser.add_argument("model_id")
    promote_model_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH
    )

    list_models_parser = subparsers.add_parser("list-models", help="List models")
    list_models_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        if args.command == "analyze":
            result = analyze_single_file(
                args.fits_file,
                model_path=args.model,
                registry_path=args.registry,
                settings_path=args.settings,
                use_default_model=not args.no_default_model,
                use_default_thresholds=not args.no_default_thresholds,
                clean_threshold=args.clean_threshold,
                contaminated_threshold=args.contaminated_threshold,
            )
            if "error" in result:
                print(f"Error: {result['error']}")
                sys.exit(1)
            print(f"File: {result['file']}")
            print(f"Quality Score: {result['quality_score']:.3f}")
            print(f"Has Streaks: {result['has_streaks']}")
            print(f"Has Clouds: {result['has_clouds']}")
            print(f"Has Saturation: {result['has_saturation']}")
            print(f"Needs Review: {result['needs_review']}")
            print(f"CV Label: {result['cv_label']}")
            print(f"Decision Label: {result['decision_label']}")
            print(f"Decision Source: {result['decision_source']}")
            print(f"Clean Threshold: {result['thresholds']['clean']:.3f}")
            print(
                "Contaminated Threshold: " f"{result['thresholds']['contaminated']:.3f}"
            )
        elif args.command == "batch":
            batch_process(
                args.input_dir,
                args.output_dir,
                action=args.action,
                model_path=args.model,
                registry_path=args.registry,
                settings_path=args.settings,
                use_default_model=not args.no_default_model,
                manifest_name=args.manifest,
                use_default_thresholds=not args.no_default_thresholds,
                clean_threshold=args.clean_threshold,
                contaminated_threshold=args.contaminated_threshold,
            )
        elif args.command == "train":
            train_model(
                args.data_dir,
                args.model_output,
                args.epochs,
                args.batch_size,
                args.train_split,
            )
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
        elif args.command == "train-from-manifest":
            train_from_manifest(
                args.manifest,
                args.model_output,
                args.dataset_dir,
                args.epochs,
                args.batch_size,
                args.reviewed_only,
            )
        elif args.command == "k8s-train":
            launch_k8s_training(args.config)
        elif args.command == "validate":
            validate_system()
        elif args.command == "review":
            review_manifest(args.manifest, open_files=args.open)
        elif args.command == "evaluate-manifest":
            report = evaluate_manifest(
                args.manifest,
                prediction=args.prediction,
                output_path=args.output,
            )
            print_evaluation_report(report)
        elif args.command == "calibrate-thresholds":
            report = calibrate_thresholds(
                args.manifests,
                step=args.step,
                min_gap=args.min_gap,
                output_path=args.output,
            )
            print_calibration_report(report)
        elif args.command == "show-thresholds":
            print_threshold_settings(args.settings)
        elif args.command == "promote-thresholds":
            if args.calibration is not None:
                if (
                    args.clean_threshold is not None
                    or args.contaminated_threshold is not None
                ):
                    raise ValueError(
                        "Use either --calibration or explicit threshold values"
                    )
                threshold_record = promote_thresholds_from_calibration(
                    args.calibration,
                    settings_path=args.settings,
                )
            else:
                if args.clean_threshold is None or args.contaminated_threshold is None:
                    raise ValueError(
                        "Explicit promotion requires --clean_threshold and "
                        "--contaminated_threshold"
                    )
                threshold_record = promote_thresholds(
                    args.clean_threshold,
                    args.contaminated_threshold,
                    settings_path=args.settings,
                )
            print(
                "Promoted thresholds: "
                f"clean={threshold_record['clean_threshold']:.3f}, "
                f"contaminated={threshold_record['contaminated_threshold']:.3f}"
            )
        elif args.command == "export-curated":
            export_manifest_path = export_curated_dataset(
                args.manifest,
                args.output_dir,
                action=args.action,
                reviewed_only=not args.include_unreviewed,
                manifest_name=args.manifest_name,
            )
            print(f"Curated dataset manifest saved to: {export_manifest_path}")
        elif args.command == "register-model":
            model_record = register_model(
                args.model_path,
                args.name,
                registry_path=args.registry,
                model_id=args.model_id,
                evaluation_path=args.evaluation,
                calibration_path=args.calibration,
                source_manifest=args.source_manifest,
                promote=args.promote,
                replace=args.replace,
            )
            print(f"Registered model: {model_record['model_id']}")
        elif args.command == "promote-model":
            model_record = promote_model(args.model_id, registry_path=args.registry)
            print(f"Promoted model: {model_record['model_id']}")
        elif args.command == "list-models":
            print_model_registry(args.registry)
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
