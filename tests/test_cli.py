"""Tests for Nebulift CLI workflow helpers."""

import json
import subprocess
import sys
from unittest.mock import patch

from nebulift.manifest import (
    batch_process,
    calibrate_thresholds,
    classify_cv_score,
    evaluate_manifest,
    export_curated_dataset,
    print_calibration_report,
    print_evaluation_report,
    review_manifest,
)
from nebulift.registry import (
    load_local_settings,
    load_model_registry,
    print_threshold_settings,
    promote_model,
    promote_thresholds,
    promote_thresholds_from_calibration,
    register_model,
    resolve_model_path,
    resolve_thresholds,
)
from nebulift.training import train_from_manifest, train_model


def _analysis(score: float) -> dict:
    return {
        "overall_quality_score": score,
        "needs_manual_review": 0.3 < score < 0.7,
        "streaks": {"has_streaks": False},
        "clouds": {"has_clouds": False},
        "saturation": {"has_saturation": False},
        "hot_pixels": {"has_hot_pixels": False},
    }


def test_classify_cv_score_three_buckets():
    """Test CV score thresholds map to the expected buckets."""
    assert classify_cv_score(0.8) == "clean"
    assert classify_cv_score(0.2) == "contaminated"
    assert classify_cv_score(0.5) == "review"


def test_cli_help_lists_release_ready_commands():
    """CLI help should expose the local-first workflow commands."""
    result = subprocess.run(
        [sys.executable, "-m", "nebulift.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "train-from-manifest" in result.stdout
    assert "evaluate-manifest" in result.stdout
    assert "calibrate-thresholds" in result.stdout
    assert "register-model" in result.stdout


def test_batch_process_defaults_to_json_report(tmp_path):
    """Batch processing should write a manifest without moving files by default."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")

    with patch("nebulift.manifest.batch_analyze_images") as mock_batch:
        mock_batch.return_value = {str(fits_file): _analysis(0.8)}
        manifest_path = batch_process(input_dir, output_dir)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["action"] == "report"
    assert manifest["summary"] == {
        "clean": 1,
        "contaminated": 0,
        "review": 0,
        "errors": 0,
    }
    assert manifest["files"][0]["decision_label"] == "clean"
    assert manifest["files"][0]["destination_path"] is None
    assert manifest["thresholds"] == {"clean": 0.7, "contaminated": 0.3}
    assert fits_file.exists()


def test_batch_process_move_action_moves_file(tmp_path):
    """Move action should move files into decision buckets."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")

    with patch("nebulift.manifest.batch_analyze_images") as mock_batch:
        mock_batch.return_value = {str(fits_file): _analysis(0.2)}
        manifest_path = batch_process(input_dir, output_dir, action="move")

    moved_file = output_dir / "contaminated" / "image.fits"
    manifest = json.loads(manifest_path.read_text())
    assert not fits_file.exists()
    assert moved_file.exists()
    assert manifest["files"][0]["decision_label"] == "contaminated"
    assert manifest["files"][0]["destination_path"] == str(moved_file)


def test_batch_process_uses_model_prediction_when_provided(tmp_path):
    """Model predictions should override CV labels when inference succeeds."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model.pth"
    input_dir.mkdir()
    model_path.write_text("placeholder")
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")

    with (
        patch("nebulift.manifest.batch_analyze_images") as mock_batch,
        patch(
            "nebulift.manifest.QualityPredictor",
        ) as mock_predictor,
    ):
        mock_batch.return_value = {str(fits_file): _analysis(0.2)}
        mock_predictor.return_value.predict_single.return_value = {
            "predicted_label": "clean",
            "confidence": 0.9,
        }
        manifest_path = batch_process(input_dir, output_dir, model_path=model_path)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"][0]["cv_label"] == "contaminated"
    assert manifest["files"][0]["decision_label"] == "clean"
    assert manifest["files"][0]["decision_source"] == "ml"


def test_batch_process_uses_promoted_default_model(tmp_path):
    """Batch processing should use the promoted registry model by default."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    registry_path = tmp_path / "registry.json"
    model_path = tmp_path / "model.pth"
    input_dir.mkdir()
    model_path.write_text("placeholder")
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")
    register_model(
        model_path,
        "Default model",
        registry_path=registry_path,
        model_id="default-model",
        promote=True,
    )

    with (
        patch("nebulift.manifest.batch_analyze_images") as mock_batch,
        patch("nebulift.manifest.QualityPredictor") as mock_predictor,
    ):
        mock_batch.return_value = {str(fits_file): _analysis(0.2)}
        mock_predictor.return_value.predict_single.return_value = {
            "predicted_label": "clean",
        }
        manifest_path = batch_process(
            input_dir,
            output_dir,
            registry_path=registry_path,
        )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_path"] == str(model_path)
    assert manifest["files"][0]["decision_source"] == "ml"


def test_batch_process_can_disable_default_model(tmp_path):
    """Batch processing can explicitly avoid the promoted default model."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    registry_path = tmp_path / "registry.json"
    model_path = tmp_path / "model.pth"
    input_dir.mkdir()
    model_path.write_text("placeholder")
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")
    register_model(
        model_path,
        "Default model",
        registry_path=registry_path,
        model_id="default-model",
        promote=True,
    )

    with (
        patch("nebulift.manifest.batch_analyze_images") as mock_batch,
        patch(
            "nebulift.manifest.QualityPredictor",
        ) as mock_predictor,
    ):
        mock_batch.return_value = {str(fits_file): _analysis(0.2)}
        manifest_path = batch_process(
            input_dir,
            output_dir,
            registry_path=registry_path,
            use_default_model=False,
        )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_path"] is None
    assert manifest["files"][0]["decision_source"] == "cv"
    mock_predictor.assert_not_called()


def test_batch_process_uses_promoted_default_thresholds(tmp_path):
    """Batch processing should use promoted threshold defaults when unset."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    settings_path = tmp_path / "local_settings.json"
    input_dir.mkdir()
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")
    promote_thresholds(0.9, 0.4, settings_path=settings_path)

    with patch("nebulift.manifest.batch_analyze_images") as mock_batch:
        mock_batch.return_value = {str(fits_file): _analysis(0.35)}
        manifest_path = batch_process(
            input_dir,
            output_dir,
            settings_path=settings_path,
        )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["thresholds"] == {"clean": 0.9, "contaminated": 0.4}
    assert manifest["files"][0]["cv_label"] == "contaminated"


def test_batch_process_can_disable_promoted_default_thresholds(tmp_path):
    """Batch processing can ignore promoted thresholds and use built-in defaults."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    settings_path = tmp_path / "local_settings.json"
    input_dir.mkdir()
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")
    promote_thresholds(0.9, 0.4, settings_path=settings_path)

    with patch("nebulift.manifest.batch_analyze_images") as mock_batch:
        mock_batch.return_value = {str(fits_file): _analysis(0.35)}
        manifest_path = batch_process(
            input_dir,
            output_dir,
            settings_path=settings_path,
            use_default_thresholds=False,
        )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["thresholds"] == {"clean": 0.7, "contaminated": 0.3}
    assert manifest["files"][0]["cv_label"] == "review"


def test_batch_process_explicit_thresholds_override_promoted_defaults(tmp_path):
    """Explicit threshold flags should override promoted local defaults."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    settings_path = tmp_path / "local_settings.json"
    input_dir.mkdir()
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")
    promote_thresholds(0.9, 0.4, settings_path=settings_path)

    with patch("nebulift.manifest.batch_analyze_images") as mock_batch:
        mock_batch.return_value = {str(fits_file): _analysis(0.35)}
        manifest_path = batch_process(
            input_dir,
            output_dir,
            settings_path=settings_path,
            clean_threshold=0.8,
            contaminated_threshold=0.2,
        )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["thresholds"] == {"clean": 0.8, "contaminated": 0.2}
    assert manifest["files"][0]["cv_label"] == "review"


def test_review_manifest_records_corrected_label(tmp_path):
    """Interactive review should persist corrected labels."""
    manifest_path = tmp_path / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "source_path": "image.fits",
                        "quality_score": 0.5,
                        "decision_label": "review",
                        "decision_source": "cv",
                        "artifacts": {},
                        "reviewed": False,
                        "corrected_label": None,
                    },
                ],
            },
        ),
    )

    with patch("builtins.input", return_value="c"):
        review_manifest(manifest_path)

    updated = json.loads(manifest_path.read_text())
    assert updated["files"][0]["reviewed"] is True
    assert updated["files"][0]["corrected_label"] == "clean"
    assert "reviewed_at" in updated


def test_review_manifest_can_open_files(tmp_path):
    """Review can optionally open each existing file with an injected opener."""
    source_file = tmp_path / "image.fits"
    source_file.write_text("placeholder")
    manifest_path = tmp_path / "batch_manifest.json"
    opened_paths = []
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_path": str(source_file),
                        "quality_score": 0.5,
                        "decision_label": "review",
                        "decision_source": "cv",
                        "artifacts": {},
                        "reviewed": False,
                        "corrected_label": None,
                    },
                ],
            },
        ),
    )

    with patch("builtins.input", return_value="s"):
        review_manifest(
            manifest_path,
            open_files=True,
            opener=lambda path: opened_paths.append(path),
        )

    assert opened_paths == [source_file]


def test_export_curated_dataset_copies_reviewed_files(tmp_path):
    """Curated export should copy reviewed files into class folders."""
    clean_file = tmp_path / "clean.fits"
    unreviewed_file = tmp_path / "unreviewed.fits"
    clean_file.write_text("clean")
    unreviewed_file.write_text("unreviewed")
    manifest_path = tmp_path / "batch_manifest.json"
    output_dir = tmp_path / "curated"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_path": str(clean_file),
                        "decision_label": "review",
                        "corrected_label": "clean",
                        "reviewed": True,
                    },
                    {
                        "source_path": str(unreviewed_file),
                        "decision_label": "contaminated",
                        "corrected_label": None,
                        "reviewed": False,
                    },
                ],
            },
        ),
    )

    export_manifest_path = export_curated_dataset(manifest_path, output_dir)

    exported_clean = output_dir / "clean" / "clean.fits"
    export_manifest = json.loads(export_manifest_path.read_text())
    assert exported_clean.exists()
    assert clean_file.exists()
    assert not (output_dir / "contaminated" / "unreviewed.fits").exists()
    assert export_manifest["summary"] == {
        "clean": 1,
        "contaminated": 0,
        "review": 0,
        "skipped": 1,
    }


def test_export_curated_dataset_can_include_unreviewed_symlinks(tmp_path):
    """Curated export can include unreviewed entries as symlinks."""
    source_file = tmp_path / "image.fits"
    source_file.write_text("placeholder")
    manifest_path = tmp_path / "batch_manifest.json"
    output_dir = tmp_path / "curated"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_path": str(source_file),
                        "decision_label": "review",
                        "corrected_label": None,
                        "reviewed": False,
                    },
                ],
            },
        ),
    )

    export_curated_dataset(
        manifest_path,
        output_dir,
        action="symlink",
        reviewed_only=False,
    )

    exported = output_dir / "review" / "image.fits"
    assert exported.is_symlink()
    assert exported.resolve() == source_file.resolve()


def test_evaluate_manifest_reports_metrics(tmp_path):
    """Manifest evaluation should compare decisions to corrected labels."""
    manifest_path = tmp_path / "batch_manifest.json"
    output_path = tmp_path / "evaluation.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "source_path": "clean.fits",
                        "cv_label": "review",
                        "decision_label": "clean",
                        "corrected_label": "clean",
                    },
                    {
                        "source_path": "bad.fits",
                        "cv_label": "clean",
                        "decision_label": "clean",
                        "corrected_label": "contaminated",
                    },
                    {
                        "source_path": "unreviewed.fits",
                        "cv_label": "review",
                        "decision_label": "review",
                        "corrected_label": None,
                    },
                ],
            },
        ),
    )

    report = evaluate_manifest(manifest_path, output_path=output_path)

    assert report["evaluated_files"] == 2
    assert report["skipped_files"] == 1
    assert report["accuracy"] == 0.5
    assert report["confusion_matrix"]["clean"]["clean"] == 1
    assert report["confusion_matrix"]["contaminated"]["clean"] == 1
    assert output_path.exists()


def test_evaluate_manifest_can_use_cv_predictions(tmp_path):
    """Manifest evaluation should support CV labels as predictions."""
    manifest_path = tmp_path / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_path": "image.fits",
                        "cv_label": "review",
                        "decision_label": "clean",
                        "corrected_label": "review",
                    },
                ],
            },
        ),
    )

    report = evaluate_manifest(manifest_path, prediction="cv")

    assert report["prediction_field"] == "cv_label"
    assert report["accuracy"] == 1.0


def test_print_evaluation_report(capsys):
    """Evaluation report printing should include key metrics."""
    print_evaluation_report(
        {
            "prediction_field": "decision_label",
            "evaluated_files": 1,
            "skipped_files": 0,
            "accuracy": 1.0,
            "per_class": {
                "clean": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            },
        },
    )

    captured = capsys.readouterr()
    assert "Accuracy: 1.000" in captured.out
    assert "clean: precision=1.000" in captured.out


def test_calibrate_thresholds_recommends_threshold_pair(tmp_path):
    """Threshold calibration should recommend values from reviewed labels."""
    manifest_path = tmp_path / "batch_manifest.json"
    output_path = tmp_path / "calibration.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {"quality_score": 0.9, "corrected_label": "clean"},
                    {"quality_score": 0.8, "corrected_label": "clean"},
                    {"quality_score": 0.5, "corrected_label": "review"},
                    {"quality_score": 0.2, "corrected_label": "contaminated"},
                    {"quality_score": 0.1, "corrected_label": "contaminated"},
                    {"quality_score": None, "corrected_label": "clean"},
                ],
            },
        ),
    )

    report = calibrate_thresholds(
        [manifest_path],
        step=0.1,
        min_gap=0.1,
        output_path=output_path,
    )

    assert report["evaluated_records"] == 5
    assert report["skipped_records"] == 1
    assert report["recommended"]["accuracy"] == 1.0
    assert report["recommended"]["macro_f1"] == 1.0
    assert output_path.exists()


def test_print_calibration_report(capsys):
    """Calibration report printing should include recommended thresholds."""
    print_calibration_report(
        {
            "evaluated_records": 5,
            "skipped_records": 1,
            "recommended": {
                "clean_threshold": 0.7,
                "contaminated_threshold": 0.3,
                "macro_f1": 0.8,
                "accuracy": 0.75,
            },
        },
    )

    captured = capsys.readouterr()
    assert "Recommended clean threshold: 0.700" in captured.out
    assert "Macro F1: 0.800" in captured.out


def test_promote_thresholds_persists_settings(tmp_path):
    """Promoting explicit thresholds should persist them in local settings."""
    settings_path = tmp_path / "local_settings.json"

    threshold_record = promote_thresholds(0.82, 0.28, settings_path=settings_path)
    settings = load_local_settings(settings_path)

    assert threshold_record["clean_threshold"] == 0.82
    assert settings["default_thresholds"]["contaminated_threshold"] == 0.28


def test_promote_thresholds_from_calibration_uses_recommended_values(tmp_path):
    """Calibration promotion should persist the recommended threshold pair."""
    settings_path = tmp_path / "local_settings.json"
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "recommended": {
                    "clean_threshold": 0.76,
                    "contaminated_threshold": 0.24,
                }
            }
        )
    )

    promote_thresholds_from_calibration(
        calibration_path,
        settings_path=settings_path,
    )
    settings = load_local_settings(settings_path)

    assert settings["default_thresholds"]["clean_threshold"] == 0.76
    assert settings["default_thresholds"]["calibration_path"] == str(calibration_path)


def test_resolve_thresholds_prefers_promoted_defaults(tmp_path):
    """Threshold resolution should use promoted defaults when explicit values are unset."""
    settings_path = tmp_path / "local_settings.json"
    promote_thresholds(0.88, 0.22, settings_path=settings_path)

    assert resolve_thresholds(settings_path=settings_path) == (0.88, 0.22)


def test_resolve_thresholds_prefers_explicit_values(tmp_path):
    """Explicit thresholds should override promoted defaults during resolution."""
    settings_path = tmp_path / "local_settings.json"
    promote_thresholds(0.88, 0.22, settings_path=settings_path)

    assert resolve_thresholds(0.8, 0.25, settings_path=settings_path) == (0.8, 0.25)


def test_print_threshold_settings(capsys, tmp_path):
    """Threshold settings output should show the promoted defaults."""
    settings_path = tmp_path / "local_settings.json"
    promote_thresholds(0.9, 0.4, settings_path=settings_path)

    print_threshold_settings(settings_path)

    captured = capsys.readouterr()
    assert "Default clean threshold: 0.900" in captured.out
    assert "Default contaminated threshold: 0.400" in captured.out


def test_train_from_manifest_calls_pipeline(tmp_path):
    """CLI helper should call manifest training pipeline with expected args."""
    manifest_path = tmp_path / "batch_manifest.json"
    model_path = tmp_path / "model.pth"
    dataset_dir = tmp_path / "dataset"
    manifest_path.write_text('{"files": []}')

    with patch(
        "nebulift.training.complete_training_pipeline_from_manifest"
    ) as mock_train:
        mock_train.return_value = {
            "model_path": str(model_path),
            "dataset_dir": str(dataset_dir),
            "dataset_stats": {
                "training_samples": 3,
                "validation_samples": 1,
                "reviewed_samples": 4,
            },
            "final_metrics": {"best_val_accuracy": 75.0},
        }
        train_from_manifest(
            manifest_path,
            model_path,
            dataset_dir,
            epochs=2,
            batch_size=4,
            reviewed_only=True,
        )

    mock_train.assert_called_once_with(
        manifest_path=manifest_path,
        model_output_path=model_path,
        dataset_output_dir=dataset_dir,
        epochs=2,
        batch_size=4,
        reviewed_only=True,
    )


def test_train_model_uses_curated_class_folders(tmp_path):
    """Train helper should consume curated class-folder FITS files."""
    data_dir = tmp_path / "curated"
    for label in ["clean", "contaminated", "review"]:
        label_dir = data_dir / label
        label_dir.mkdir(parents=True)
        (label_dir / f"{label}.fits").write_text("placeholder")
    model_output = tmp_path / "model.pth"

    with (
        patch("nebulift.training.AstroImageDataset") as mock_dataset,
        patch("nebulift.training.DataLoader") as mock_loader,
        patch("nebulift.training.AstroQualityClassifier") as mock_model,
        patch("nebulift.training.ModelTrainer") as mock_trainer,
    ):
        mock_dataset.side_effect = [
            type("Dataset", (), {"__len__": lambda self: 2})(),
            type("Dataset", (), {"__len__": lambda self: 1})(),
        ]
        train_model(data_dir, model_output, epochs=2, batch_size=2, train_split=0.67)

    mock_model.assert_called_once_with(num_classes=3, pretrained=False)
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once_with(model_output)
    assert mock_dataset.call_count == 2
    assert mock_loader.call_count == 2


def test_register_promote_and_resolve_model(tmp_path):
    """Model registry should register, promote, and resolve defaults."""
    registry_path = tmp_path / "model_registry.json"
    model_path = tmp_path / "classifier.pth"
    evaluation_path = tmp_path / "evaluation.json"
    model_path.write_text("placeholder")
    evaluation_path.write_text(json.dumps({"accuracy": 0.9}))

    model_record = register_model(
        model_path,
        "Classifier",
        registry_path=registry_path,
        model_id="classifier-v1",
        evaluation_path=evaluation_path,
    )
    promoted_record = promote_model("classifier-v1", registry_path=registry_path)
    registry = load_model_registry(registry_path)

    assert model_record["model_id"] == "classifier-v1"
    assert promoted_record["model_path"] == str(model_path)
    assert registry["default_model_id"] == "classifier-v1"
    assert registry["models"]["classifier-v1"]["evaluation"] == {"accuracy": 0.9}
    assert resolve_model_path(registry_path=registry_path) == model_path


def test_resolve_model_path_prefers_explicit_model(tmp_path):
    """Explicit model paths should take precedence over registry defaults."""
    explicit_model = tmp_path / "explicit.pth"
    explicit_model.write_text("placeholder")

    assert resolve_model_path(explicit_model) == explicit_model


def test_register_model_prevents_duplicate_ids(tmp_path):
    """Registering the same model id twice should require replacement."""
    registry_path = tmp_path / "model_registry.json"
    model_path = tmp_path / "classifier.pth"
    model_path.write_text("placeholder")
    register_model(
        model_path,
        "Classifier",
        registry_path=registry_path,
        model_id="classifier-v1",
    )

    try:
        register_model(
            model_path,
            "Classifier",
            registry_path=registry_path,
            model_id="classifier-v1",
        )
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected duplicate model id to fail")
