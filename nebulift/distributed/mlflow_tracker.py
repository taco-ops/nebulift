"""
MLflow experiment-tracking integration for the in-cluster trainer.

Design goals
------------
- **Optional dependency.** ``mlflow`` is declared in the ``tracking`` extra
  and imported lazily inside :meth:`MLflowTracker.__enter__` so a local
  CLI install without the extra still works.
- **Graceful degradation.** Any missing config, missing dependency, or
  failed call collapses the tracker to a no-op. Training never fails
  because experiment tracking failed; the worst case is a missing run
  in MLflow with a logged exception.
- **Rank-aware.** Only rank 0 in a distributed job creates a run and
  emits metrics/artifacts. Non-zero ranks receive a no-op tracker from
  :func:`tracker_from_env`, which keeps the call sites identical across
  ranks without ``if rank == 0`` ladders.

The tracker is intentionally **not** a generic mlflow facade; it exposes
only the surface the trainer needs (params at start, metrics per epoch,
final checkpoint as an artifact). Anything richer should live in a
dedicated module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from types import ModuleType

logger = logging.getLogger(__name__)

# MLflow's tracking server caps string params at 6000 characters. Truncate
# locally so long config dumps (CLI argv, resolved paths, etc.) don't
# trigger a 400 from the server and abort the run.
_PARAM_MAX_LEN = 6000

# Env vars consumed by tracker_from_env(). Kept in one place so the
# documentation and Kubernetes manifests stay in sync.
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"
ENV_RUN_NAME = "MLFLOW_RUN_NAME"

# Environment-derived tags attached to every run. The keys are stable
# across runs so MLflow's UI search ("tags.nebulift.environment = ...")
# works. Empty values are skipped to avoid noisy "" entries.
_TAG_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("nebulift.environment", "ENVIRONMENT"),
    ("nebulift.image_tag", "IMAGE_TAG"),
    ("nebulift.commit_sha", "COMMIT_SHA"),
    ("nebulift.pod_name", "POD_NAME"),
    ("nebulift.pod_namespace", "POD_NAMESPACE"),
    ("nebulift.job_name", "JOB_NAME"),
    ("nebulift.world_size", "WORLD_SIZE"),
)


def _coerce_param(value: Any) -> str:
    """Render an arbitrary value as an mlflow-safe param string.

    MLflow params must be strings ≤ 6000 chars. Non-string inputs go
    through :func:`repr` so dataclasses / paths / lists round-trip in a
    readable form; oversized strings are truncated with an explicit
    ``...`` marker so it's obvious in the UI that the value was cut.
    """
    text = value if isinstance(value, str) else repr(value)
    if len(text) > _PARAM_MAX_LEN:
        suffix = "...[truncated]"
        text = text[: _PARAM_MAX_LEN - len(suffix)] + suffix
    return text


class MLflowTracker:
    """Context-managed wrapper around an MLflow run.

    Construct with a fully-resolved tracking URI + experiment name. Use
    as a context manager so a successful exit ends the run with status
    ``FINISHED`` and an exception ends it with ``FAILED``::

        with MLflowTracker(uri, "nebulift-development") as tracker:
            tracker.log_params({"epochs": 10})
            tracker.log_metrics({"val_accuracy": 88.4}, step=1)
            tracker.log_artifact(checkpoint_path)

    If ``tracking_uri`` is falsy the tracker becomes a no-op immediately;
    callers don't need to branch on configuration.
    """

    def __init__(
        self,
        tracking_uri: Optional[str],
        experiment_name: Optional[str],
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._tracking_uri = tracking_uri or None
        self._experiment_name = experiment_name or None
        self._run_name = run_name or None
        self._tags = dict(tags) if tags else {}
        self._enabled = False
        self._mlflow: Optional["ModuleType"] = None
        self._active_run: Any = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "MLflowTracker":
        """Resolve the mlflow client and start a run.

        Any failure (missing dependency, unreachable server, invalid
        experiment name) is logged and leaves the tracker in disabled
        state. The context still enters successfully so the caller's
        ``with`` block runs as a normal no-op.
        """
        if self._tracking_uri is None:
            logger.info(
                "MLflow tracker disabled: no tracking URI provided "
                "(set %s to enable).",
                ENV_TRACKING_URI,
            )
            return self

        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "MLflow tracker disabled: 'mlflow' package not installed. "
                "Install with the 'tracking' extra to enable experiment tracking.",
            )
            return self

        self._mlflow = mlflow
        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            if self._experiment_name:
                mlflow.set_experiment(self._experiment_name)
            self._active_run = mlflow.start_run(
                run_name=self._run_name,
                tags=self._tags or None,
            )
        except Exception:
            logger.exception(
                "MLflow tracker disabled: failed to start run against %s",
                self._tracking_uri,
            )
            self._active_run = None
            self._mlflow = None
            return self

        self._enabled = True
        logger.info(
            "MLflow tracking enabled: uri=%s experiment=%s run_name=%s",
            self._tracking_uri,
            self._experiment_name,
            self._run_name,
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Any,
    ) -> None:
        """End the run, marking it ``FAILED`` if the with-block raised.

        The exit never raises: tracking-server errors during teardown
        must not mask the real exception (if any) propagating from the
        training body.
        """
        if not self._enabled or self._mlflow is None:
            return
        status = "FAILED" if exc_type is not None else "FINISHED"
        try:
            self._mlflow.end_run(status=status)
        except Exception:
            logger.exception("MLflow end_run failed (status=%s)", status)
        finally:
            self._enabled = False
            self._active_run = None
            self._mlflow = None

    # ------------------------------------------------------------------
    # Public surface used by the trainer
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """``True`` once :meth:`__enter__` has successfully started a run."""
        return self._enabled

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record run-wide parameters (logged once at run start).

        Values are coerced to MLflow-safe strings via :func:`_coerce_param`.
        Errors are swallowed and logged so a missing param key cannot
        bring down the training Job.
        """
        if not self._enabled or self._mlflow is None or not params:
            return
        safe = {key: _coerce_param(value) for key, value in params.items()}
        try:
            self._mlflow.log_params(safe)
        except Exception:
            logger.exception("MLflow log_params failed (keys=%s)", sorted(safe))

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int]) -> None:
        """Record a batch of scalar metrics for a given step (epoch).

        Non-finite or non-numeric values are dropped silently because
        MLflow's REST API rejects them; logging would only add noise.
        """
        if not self._enabled or self._mlflow is None or not metrics:
            return
        cleaned: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric != numeric or numeric in (float("inf"), float("-inf")):
                # NaN / inf: MLflow rejects these; skip to avoid 400s.
                continue
            cleaned[key] = numeric
        if not cleaned:
            return
        try:
            self._mlflow.log_metrics(cleaned, step=step)
        except Exception:
            logger.exception(
                "MLflow log_metrics failed (keys=%s, step=%s)",
                sorted(cleaned),
                step,
            )

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        """Upload a single file (typically the final checkpoint) to the run.

        Missing files are skipped with a warning rather than treated as
        an error: the trainer may legitimately call this on a non-rank-0
        machine where no checkpoint was written.
        """
        if not self._enabled or self._mlflow is None:
            return
        if not path.exists():
            logger.warning("MLflow log_artifact skipped: %s does not exist", path)
            return
        try:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception:
            logger.exception("MLflow log_artifact failed for %s", path)


def tracker_from_env(*, rank: int = 0) -> MLflowTracker:
    """Build an :class:`MLflowTracker` from the process environment.

    Non-zero ranks always get a disabled tracker so distributed jobs
    don't produce N parallel MLflow runs. Rank 0 reads
    ``MLFLOW_TRACKING_URI`` / ``MLFLOW_EXPERIMENT_NAME`` /
    ``MLFLOW_RUN_NAME`` and attaches the standard ``nebulift.*`` tag set
    derived from the Kubernetes downward-API env (POD_NAME, JOB_NAME,
    WORLD_SIZE, etc.). Empty tag values are dropped.

    The factory itself never raises; configuration failures surface
    inside :meth:`MLflowTracker.__enter__` so the trainer's with-block
    is the single failure boundary.
    """
    if rank != 0:
        return MLflowTracker(tracking_uri=None, experiment_name=None)

    tracking_uri = os.environ.get(ENV_TRACKING_URI)
    experiment_name = os.environ.get(ENV_EXPERIMENT_NAME)
    # Default run name to the pod name so MLflow's UI shows a stable,
    # human-readable identifier instead of an auto-generated nonce. Fall
    # back to mlflow's auto-naming when POD_NAME is unset (local runs).
    run_name = os.environ.get(ENV_RUN_NAME) or os.environ.get("POD_NAME")

    tags: dict[str, str] = {}
    for tag_key, env_var in _TAG_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            tags[tag_key] = value

    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
    )
