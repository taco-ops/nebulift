"""Unit tests for the MLflow experiment-tracking integration.

These tests mock the ``mlflow`` package at module boundaries so they
exercise the tracker without requiring the optional ``tracking`` extra
to be installed on the test runner. They cover the three modes the
trainer relies on: tracker disabled, tracker enabled-and-successful,
and tracker enabled-but-mlflow-misbehaves (errors must be swallowed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from nebulift.distributed import mlflow_tracker
from nebulift.distributed.mlflow_tracker import (
    ENV_EXPERIMENT_NAME,
    ENV_RUN_NAME,
    ENV_TRACKING_URI,
    MLflowTracker,
    _coerce_param,
    tracker_from_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _install_fake_mlflow() -> MagicMock:
    """Install a MagicMock as ``mlflow`` in ``sys.modules``.

    Returns the mock so individual tests can assert calls. The fixture
    is reset between tests by ``mlflow_module`` to avoid leaking state.
    """
    fake = MagicMock(name="mlflow_module", spec=ModuleType("mlflow"))
    # ``mlflow`` is consumed dynamically inside ``__enter__`` via
    # ``import mlflow``; we make ``spec`` permissive by attaching the
    # specific attributes the tracker calls.
    fake.set_tracking_uri = MagicMock(name="set_tracking_uri")
    fake.set_experiment = MagicMock(name="set_experiment")
    fake.start_run = MagicMock(name="start_run", return_value=MagicMock())
    fake.end_run = MagicMock(name="end_run")
    fake.log_params = MagicMock(name="log_params")
    fake.log_metrics = MagicMock(name="log_metrics")
    fake.log_artifact = MagicMock(name="log_artifact")
    sys.modules["mlflow"] = fake
    return fake


@pytest.fixture
def mlflow_module() -> MagicMock:
    """Pytest fixture wrapping ``_install_fake_mlflow`` with cleanup."""
    fake = _install_fake_mlflow()
    yield fake
    sys.modules.pop("mlflow", None)


# ---------------------------------------------------------------------------
# _coerce_param
# ---------------------------------------------------------------------------
def test_coerce_param_passes_short_strings_through() -> None:
    """Short string params survive untouched (no repr quoting)."""
    assert _coerce_param("hello") == "hello"


def test_coerce_param_uses_repr_for_non_strings() -> None:
    """Non-string values go through ``repr`` so dataclasses/paths read well."""
    assert _coerce_param(42) == "42"
    assert _coerce_param([1, 2]) == "[1, 2]"


def test_coerce_param_truncates_with_marker() -> None:
    """Strings over 6000 chars are truncated and tagged so the cut is visible in MLflow."""
    huge = "x" * 7000
    coerced = _coerce_param(huge)
    assert len(coerced) <= 6000
    assert coerced.endswith("...[truncated]")


# ---------------------------------------------------------------------------
# MLflowTracker — disabled paths
# ---------------------------------------------------------------------------
def test_tracker_disabled_without_uri_is_noop() -> None:
    """Missing tracking URI yields a tracker that no-ops every method."""
    tracker = MLflowTracker(tracking_uri=None, experiment_name="anything")
    with tracker as ctx:
        assert ctx is tracker
        assert ctx.enabled is False
        # All public methods must be safe to call.
        ctx.log_params({"a": 1})
        ctx.log_metrics({"loss": 0.1}, step=1)
        ctx.log_artifact(Path("/does/not/exist"))


def test_tracker_disabled_when_mlflow_import_fails() -> None:
    """Missing 'mlflow' package leaves the tracker disabled, not raising."""
    # Ensure mlflow is not importable for the duration of this test.
    with patch.dict(sys.modules, {"mlflow": None}):
        tracker = MLflowTracker(
            tracking_uri="http://mlflow.local:5000",
            experiment_name="exp",
        )
        with tracker as ctx:
            assert ctx.enabled is False


def test_tracker_disabled_when_start_run_raises(mlflow_module: MagicMock) -> None:
    """A failing ``start_run`` collapses the tracker to no-op, not propagated."""
    mlflow_module.start_run.side_effect = RuntimeError("server down")
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with tracker as ctx:
        assert ctx.enabled is False
        # Subsequent calls remain safe.
        ctx.log_metrics({"loss": 0.1}, step=1)
    # We must not call end_run when start_run failed; otherwise mlflow
    # would error out trying to close a non-existent active run.
    mlflow_module.end_run.assert_not_called()


# ---------------------------------------------------------------------------
# MLflowTracker — enabled happy path
# ---------------------------------------------------------------------------
def test_tracker_enabled_sets_uri_and_experiment(mlflow_module: MagicMock) -> None:
    """Entering the context configures the mlflow client and starts a run."""
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="nebulift-development",
        run_name="pod-xyz",
        tags={"nebulift.environment": "development"},
    )
    with tracker as ctx:
        assert ctx.enabled is True
    mlflow_module.set_tracking_uri.assert_called_once_with("http://mlflow.local:5000")
    mlflow_module.set_experiment.assert_called_once_with("nebulift-development")
    mlflow_module.start_run.assert_called_once_with(
        run_name="pod-xyz",
        tags={"nebulift.environment": "development"},
    )
    mlflow_module.end_run.assert_called_once_with(status="FINISHED")


def test_tracker_ends_run_with_failed_on_exception(mlflow_module: MagicMock) -> None:
    """Exceptions inside the with-block end the run as ``FAILED`` and propagate."""
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with pytest.raises(ValueError, match="boom"):
        with tracker:
            raise ValueError("boom")
    mlflow_module.end_run.assert_called_once_with(status="FAILED")


def test_tracker_log_params_coerces_values(mlflow_module: MagicMock) -> None:
    """``log_params`` stringifies every value before handing off to mlflow."""
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with tracker as ctx:
        ctx.log_params({"epochs": 10, "lr": 0.001, "name": "alpha"})
    mlflow_module.log_params.assert_called_once_with(
        {"epochs": "10", "lr": "0.001", "name": "alpha"},
    )


def test_tracker_log_metrics_drops_nan_and_inf(mlflow_module: MagicMock) -> None:
    """NaN/inf values are dropped because the MLflow REST API rejects them."""
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with tracker as ctx:
        ctx.log_metrics(
            {
                "loss": 0.25,
                "diverged": float("nan"),
                "explode": float("inf"),
                "bad": "not-a-number",
            },
            step=3,
        )
    mlflow_module.log_metrics.assert_called_once_with({"loss": 0.25}, step=3)


def test_tracker_log_metrics_skips_when_only_invalid(mlflow_module: MagicMock) -> None:
    """If every metric is filtered out, the underlying client is not called."""
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with tracker as ctx:
        ctx.log_metrics({"nan": float("nan")}, step=1)
    mlflow_module.log_metrics.assert_not_called()


def test_tracker_log_artifact_uploads_existing_file(
    mlflow_module: MagicMock, tmp_path: Path
) -> None:
    """An existing checkpoint file is forwarded to ``mlflow.log_artifact``."""
    artifact = tmp_path / "model.pth"
    artifact.write_bytes(b"weights")
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with tracker as ctx:
        ctx.log_artifact(artifact, artifact_path="model")
    mlflow_module.log_artifact.assert_called_once_with(
        str(artifact), artifact_path="model"
    )


def test_tracker_log_artifact_skips_missing_file(
    mlflow_module: MagicMock, tmp_path: Path
) -> None:
    """Missing artifacts are skipped (logged as a warning), not uploaded."""
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with tracker as ctx:
        ctx.log_artifact(tmp_path / "absent.pth")
    mlflow_module.log_artifact.assert_not_called()


def test_tracker_swallows_client_errors_during_logging(
    mlflow_module: MagicMock,
) -> None:
    """A misbehaving mlflow client must not propagate errors into training."""
    mlflow_module.log_params.side_effect = RuntimeError("server hiccup")
    mlflow_module.log_metrics.side_effect = RuntimeError("server hiccup")
    mlflow_module.log_artifact.side_effect = RuntimeError("server hiccup")
    tracker = MLflowTracker(
        tracking_uri="http://mlflow.local:5000",
        experiment_name="exp",
    )
    with tracker as ctx:
        # None of these should raise; the trainer is expected to keep going.
        ctx.log_params({"k": "v"})
        ctx.log_metrics({"loss": 0.5}, step=0)


# ---------------------------------------------------------------------------
# tracker_from_env
# ---------------------------------------------------------------------------
def test_tracker_from_env_returns_noop_on_non_zero_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker ranks must always get a disabled tracker to prevent duplicate runs."""
    monkeypatch.setenv(ENV_TRACKING_URI, "http://mlflow.local:5000")
    monkeypatch.setenv(ENV_EXPERIMENT_NAME, "exp")
    tracker = tracker_from_env(rank=1)
    # The tracker is disabled before entering and stays disabled after.
    with tracker as ctx:
        assert ctx.enabled is False


def test_tracker_from_env_uses_pod_name_as_default_run_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MLFLOW_RUN_NAME is unset, the run name falls back to POD_NAME."""
    monkeypatch.setenv(ENV_TRACKING_URI, "http://mlflow.local:5000")
    monkeypatch.setenv(ENV_EXPERIMENT_NAME, "exp")
    monkeypatch.delenv(ENV_RUN_NAME, raising=False)
    monkeypatch.setenv("POD_NAME", "nebulift-training-2-abc")

    tracker = tracker_from_env(rank=0)
    assert tracker._run_name == "nebulift-training-2-abc"


def test_tracker_from_env_collects_kubernetes_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standard ``nebulift.*`` tags are populated from downward-API env vars."""
    monkeypatch.setenv(ENV_TRACKING_URI, "http://mlflow.local:5000")
    monkeypatch.setenv(ENV_EXPERIMENT_NAME, "exp")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("POD_NAME", "pod-1")
    monkeypatch.setenv("POD_NAMESPACE", "nebulift")
    monkeypatch.setenv("WORLD_SIZE", "4")
    # Intentionally unset to verify empty tags are dropped (not "" entries).
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    monkeypatch.delenv("COMMIT_SHA", raising=False)
    monkeypatch.delenv("JOB_NAME", raising=False)

    tracker = tracker_from_env(rank=0)
    assert tracker._tags == {
        "nebulift.environment": "development",
        "nebulift.pod_name": "pod-1",
        "nebulift.pod_namespace": "nebulift",
        "nebulift.world_size": "4",
    }


def test_tracker_from_env_disabled_without_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``MLFLOW_TRACKING_URI``, rank 0 still gets a disabled tracker."""
    monkeypatch.delenv(ENV_TRACKING_URI, raising=False)
    tracker = tracker_from_env(rank=0)
    with tracker as ctx:
        assert ctx.enabled is False


def test_tracker_from_env_constants_match_module_exports() -> None:
    """The env-var constants are part of the module's public surface for callers."""
    assert mlflow_tracker.ENV_TRACKING_URI == "MLFLOW_TRACKING_URI"
    assert mlflow_tracker.ENV_EXPERIMENT_NAME == "MLFLOW_EXPERIMENT_NAME"
    assert mlflow_tracker.ENV_RUN_NAME == "MLFLOW_RUN_NAME"
