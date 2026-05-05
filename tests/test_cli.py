"""Tests for Nebulift CLI workflow helpers."""

import json
from unittest.mock import patch

from nebulift.cli import batch_process, classify_cv_score, review_manifest


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


def test_batch_process_defaults_to_json_report(tmp_path):
    """Batch processing should write a manifest without moving files by default."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")

    with patch("nebulift.cli.batch_analyze_images") as mock_batch:
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
    assert fits_file.exists()


def test_batch_process_move_action_moves_file(tmp_path):
    """Move action should move files into decision buckets."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fits_file = input_dir / "image.fits"
    fits_file.write_text("placeholder")

    with patch("nebulift.cli.batch_analyze_images") as mock_batch:
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
        patch("nebulift.cli.batch_analyze_images") as mock_batch,
        patch(
            "nebulift.cli.QualityPredictor",
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
