"""Manifest, review, evaluation, export, and calibration helpers."""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .cv_prefilter import ArtifactDetector, batch_analyze_images
from .fits_processor import FITSProcessor
from .ml_model import QualityPredictor
from .registry import (
    DEFAULT_LOCAL_SETTINGS_PATH,
    DEFAULT_REGISTRY_PATH,
    resolve_model_path,
    resolve_thresholds,
)

VALID_LABELS = {"clean", "contaminated", "review"}


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
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    settings_path: Path = DEFAULT_LOCAL_SETTINGS_PATH,
    use_default_model: bool = True,
    use_default_thresholds: bool = True,
    clean_threshold: Optional[float] = None,
    contaminated_threshold: Optional[float] = None,
) -> dict[str, Any]:
    """Analyze a single FITS file for quality."""
    processor = FITSProcessor()
    detector = ArtifactDetector()
    resolved_clean_threshold, resolved_contaminated_threshold = resolve_thresholds(
        clean_threshold=clean_threshold,
        contaminated_threshold=contaminated_threshold,
        settings_path=settings_path,
        use_default_thresholds=use_default_thresholds,
    )

    fits_data = processor.load_fits_file(fits_path)
    if not fits_data:
        return {"error": f"Failed to load FITS file: {fits_path}"}

    normalized = processor.normalize_image(fits_data["image_data"])
    analysis = detector.comprehensive_analysis(normalized)
    quality_score = float(analysis["overall_quality_score"])
    cv_label = classify_cv_score(
        quality_score,
        clean_threshold=resolved_clean_threshold,
        contaminated_threshold=resolved_contaminated_threshold,
    )

    ml_prediction = None
    decision_label = cv_label
    decision_source = "cv"
    active_model_path = resolve_model_path(model_path, registry_path, use_default_model)
    if active_model_path is not None:
        predictor = QualityPredictor(str(active_model_path))
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
        "model_path": str(active_model_path) if active_model_path else None,
        "thresholds": {
            "clean": resolved_clean_threshold,
            "contaminated": resolved_contaminated_threshold,
        },
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
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    settings_path: Path = DEFAULT_LOCAL_SETTINGS_PATH,
    use_default_model: bool = True,
    manifest_name: str = "batch_manifest.json",
    use_default_thresholds: bool = True,
    clean_threshold: Optional[float] = None,
    contaminated_threshold: Optional[float] = None,
) -> Path:
    """Batch process FITS files and write a JSON manifest."""
    if action not in {"report", "move"}:
        raise ValueError("action must be 'report' or 'move'")

    detector = ArtifactDetector()
    processor = FITSProcessor()
    output_dir.mkdir(parents=True, exist_ok=True)
    active_model_path = resolve_model_path(model_path, registry_path, use_default_model)
    resolved_clean_threshold, resolved_contaminated_threshold = resolve_thresholds(
        clean_threshold=clean_threshold,
        contaminated_threshold=contaminated_threshold,
        settings_path=settings_path,
        use_default_thresholds=use_default_thresholds,
    )

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
                    "model_path": str(active_model_path) if active_model_path else None,
                    "thresholds": {
                        "clean": resolved_clean_threshold,
                        "contaminated": resolved_contaminated_threshold,
                    },
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
    results = batch_analyze_images([str(f) for f in fits_files], detector, processor)
    predictor = QualityPredictor(str(active_model_path)) if active_model_path else None
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
            clean_threshold=resolved_clean_threshold,
            contaminated_threshold=resolved_contaminated_threshold,
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
                    clean_threshold=resolved_clean_threshold,
                    contaminated_threshold=resolved_contaminated_threshold,
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
        "model_path": str(active_model_path) if active_model_path else None,
        "thresholds": {
            "clean": resolved_clean_threshold,
            "contaminated": resolved_contaminated_threshold,
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


def _entry_file_path(entry: dict[str, Any]) -> Optional[Path]:
    path_value = entry.get("destination_path") or entry.get("source_path")
    if not path_value:
        return None
    return Path(path_value)


def open_file_in_viewer(file_path: Path) -> None:
    """Open a file with the platform default viewer."""
    if sys.platform == "darwin":
        subprocess.run(["open", str(file_path)], check=False)  # nosec B607
    elif sys.platform.startswith("win"):
        subprocess.run(
            ["cmd", "/c", "start", "", str(file_path)], check=False
        )  # nosec B607
    else:
        subprocess.run(["xdg-open", str(file_path)], check=False)  # nosec B607


def review_manifest(
    manifest_path: Path,
    open_files: bool = False,
    opener: Optional[Any] = None,
) -> None:
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
        file_path = _entry_file_path(entry)
        print()
        print(f"[{index}/{len(files)}] {file_path or entry.get('source_path')}")
        print(f"Quality score: {entry.get('quality_score')}")
        print(
            f"Decision: {entry.get('decision_label')} ({entry.get('decision_source')})"
        )
        print(f"Current label: {current_label}")
        print(f"Artifacts: {entry.get('artifacts')}")

        if open_files and file_path and file_path.exists():
            (opener or open_file_in_viewer)(file_path)

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


def _unique_destination(destination_dir: Path, file_name: str) -> Path:
    destination = destination_dir / file_name
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def export_curated_dataset(
    manifest_path: Path,
    output_dir: Path,
    action: str = "copy",
    reviewed_only: bool = True,
    manifest_name: str = "curated_manifest.json",
) -> Path:
    """Export manifest labels into class folders for curated datasets."""
    if action not in {"copy", "symlink", "move"}:
        raise ValueError("action must be 'copy', 'symlink', or 'move'")

    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_entries = []
    skipped = 0

    for entry in manifest.get("files", []):
        if entry.get("error"):
            skipped += 1
            continue
        if reviewed_only and not entry.get("reviewed"):
            skipped += 1
            continue

        label = entry.get("corrected_label") or entry.get("decision_label")
        if label not in VALID_LABELS:
            skipped += 1
            continue

        source_path = _entry_file_path(entry)
        if source_path is None or not source_path.exists():
            skipped += 1
            continue

        destination_dir = output_dir / str(label)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = _unique_destination(destination_dir, source_path.name)

        if action == "copy":
            shutil.copy2(source_path, destination_path)
        elif action == "symlink":
            destination_path.symlink_to(source_path.resolve())
        else:
            shutil.move(str(source_path), str(destination_path))

        exported_entries.append(
            {
                "source_path": str(source_path),
                "exported_path": str(destination_path),
                "label": label,
                "reviewed": bool(entry.get("reviewed")),
                "action": action,
            },
        )

    summary = {"clean": 0, "contaminated": 0, "review": 0, "skipped": skipped}
    for entry in exported_entries:
        summary[str(entry["label"])] += 1

    export_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "action": action,
        "reviewed_only": reviewed_only,
        "summary": summary,
        "files": exported_entries,
    }
    export_manifest_path = output_dir / manifest_name
    export_manifest_path.write_text(json.dumps(export_manifest, indent=2))
    return export_manifest_path


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _empty_confusion_matrix(labels: list[str]) -> dict[str, dict[str, int]]:
    return {actual: {predicted: 0 for predicted in labels} for actual in labels}


def _metrics_from_confusion_matrix(
    confusion_matrix: dict[str, dict[str, int]],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    per_class = {}
    for label in labels:
        true_positive = confusion_matrix[label][label]
        false_negative = sum(confusion_matrix[label].values()) - true_positive
        false_positive = (
            sum(confusion_matrix[actual][label] for actual in labels) - true_positive
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1_score = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }
    return per_class


def _macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    if not per_class:
        return 0.0
    return sum(metrics["f1"] for metrics in per_class.values()) / len(per_class)


def evaluate_manifest(
    manifest_path: Path,
    prediction: str = "decision",
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Evaluate manifest predictions against reviewed corrected labels."""
    if prediction not in {"decision", "cv"}:
        raise ValueError("prediction must be 'decision' or 'cv'")

    manifest = json.loads(manifest_path.read_text())
    prediction_field = "decision_label" if prediction == "decision" else "cv_label"
    labels = ["clean", "contaminated", "review"]
    confusion_matrix = _empty_confusion_matrix(labels)
    skipped_files = 0
    evaluated_files = 0
    correct = 0

    for entry in manifest.get("files", []):
        actual_label = entry.get("corrected_label")
        predicted_label = entry.get(prediction_field)

        if actual_label not in labels or predicted_label not in labels:
            skipped_files += 1
            continue

        evaluated_files += 1
        confusion_matrix[actual_label][predicted_label] += 1
        if actual_label == predicted_label:
            correct += 1

    per_class = _metrics_from_confusion_matrix(confusion_matrix, labels)

    report = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "prediction_field": prediction_field,
        "total_files": len(manifest.get("files", [])),
        "evaluated_files": evaluated_files,
        "skipped_files": skipped_files,
        "accuracy": _safe_divide(correct, evaluated_files),
        "confusion_matrix": confusion_matrix,
        "per_class": per_class,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    return report


def _reviewed_score_records(
    manifest_paths: list[Path],
) -> tuple[list[dict[str, Any]], int]:
    records = []
    skipped = 0
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest.get("files", []):
            actual_label = entry.get("corrected_label")
            quality_score = entry.get("quality_score")
            if actual_label not in VALID_LABELS or quality_score is None:
                skipped += 1
                continue
            records.append(
                {
                    "manifest_path": str(manifest_path),
                    "source_path": entry.get("source_path"),
                    "quality_score": float(quality_score),
                    "actual_label": actual_label,
                },
            )
    return records, skipped


def _threshold_candidates(step: float) -> list[float]:
    count = int(round(1.0 / step))
    return [round(index * step, 6) for index in range(count + 1)]


def calibrate_thresholds(
    manifest_paths: list[Path],
    step: float = 0.05,
    min_gap: float = 0.1,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Calibrate CV clean/contaminated thresholds from reviewed manifests."""
    if not 0.0 < step <= 0.5:
        raise ValueError("step must be greater than 0.0 and no more than 0.5")
    if not 0.0 <= min_gap < 1.0:
        raise ValueError("min_gap must be between 0.0 and 1.0")

    labels = ["clean", "contaminated", "review"]
    records, skipped_records = _reviewed_score_records(manifest_paths)
    if not records:
        raise ValueError("No reviewed manifest entries with quality scores were found")

    best_result = None
    all_results = []
    candidates = _threshold_candidates(step)
    for contaminated_threshold in candidates:
        for clean_threshold in candidates:
            if contaminated_threshold + min_gap > clean_threshold:
                continue

            confusion_matrix = _empty_confusion_matrix(labels)
            correct = 0
            for record in records:
                predicted_label = classify_cv_score(
                    record["quality_score"],
                    clean_threshold=clean_threshold,
                    contaminated_threshold=contaminated_threshold,
                )
                actual_label = str(record["actual_label"])
                confusion_matrix[actual_label][predicted_label] += 1
                if actual_label == predicted_label:
                    correct += 1

            per_class = _metrics_from_confusion_matrix(confusion_matrix, labels)
            result = {
                "clean_threshold": clean_threshold,
                "contaminated_threshold": contaminated_threshold,
                "accuracy": _safe_divide(correct, len(records)),
                "macro_f1": _macro_f1(per_class),
                "confusion_matrix": confusion_matrix,
                "per_class": per_class,
            }
            all_results.append(result)
            if best_result is None or (
                result["macro_f1"],
                result["accuracy"],
            ) > (
                best_result["macro_f1"],
                best_result["accuracy"],
            ):
                best_result = result

    assert best_result is not None
    top_results = sorted(
        all_results,
        key=lambda item: (item["macro_f1"], item["accuracy"]),
        reverse=True,
    )[:10]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_paths": [str(path) for path in manifest_paths],
        "step": step,
        "min_gap": min_gap,
        "evaluated_records": len(records),
        "skipped_records": skipped_records,
        "recommended": best_result,
        "top_results": top_results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    return report


def print_calibration_report(report: dict[str, Any]) -> None:
    """Print a concise threshold calibration summary."""
    recommended = report["recommended"]
    print(f"Evaluated records: {report['evaluated_records']}")
    print(f"Skipped records: {report['skipped_records']}")
    print(f"Recommended clean threshold: {recommended['clean_threshold']:.3f}")
    print(
        "Recommended contaminated threshold: "
        f"{recommended['contaminated_threshold']:.3f}",
    )
    print(f"Macro F1: {recommended['macro_f1']:.3f}")
    print(f"Accuracy: {recommended['accuracy']:.3f}")


def print_evaluation_report(report: dict[str, Any]) -> None:
    """Print a concise manifest evaluation summary."""
    print(f"Prediction field: {report['prediction_field']}")
    print(f"Evaluated files: {report['evaluated_files']}")
    print(f"Skipped files: {report['skipped_files']}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print("Per-class metrics:")
    for label, metrics in report["per_class"].items():
        print(
            f"  {label}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}",
        )
