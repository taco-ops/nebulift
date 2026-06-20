# Phase 1b-1: Trainer-side MLflow Integration

## Summary

Phase 1b-1 wires the in-cluster distributed trainer into the Phase 1a MLflow
tracking server. Every distributed training Job now creates an MLflow run from
rank 0 that captures the hyperparameters, per-epoch metrics, and the final
checkpoint artifact. The platform deployed in Phase 1a (MLflow + MinIO behind
the `nebulift-platform` Gateway) is unchanged; this phase is additive on the
trainer side only.

The integration is designed to fail open: if MLflow is unreachable, the
`tracking` extra is missing, or the tracking server rejects a call, the
training Job still completes successfully and writes its checkpoint to NFS as
before. The only visible effect of a tracking failure is a missing run in the
MLflow UI plus a logged exception in the pod's stdout.

## Scope and Non-Goals

In scope:

- New `nebulift.distributed.mlflow_tracker` module providing `MLflowTracker`
  (context manager) and `tracker_from_env()` (rank-aware factory).
- New optional `EpochCallback` hook on `ModelTrainer.train()` that emits a
  per-epoch metric dict.
- Integration in `nebulift.distributed.k8s_trainer.main()`: rank 0 starts a
  run, logs params, streams per-epoch metrics, and uploads the final
  checkpoint as the `model/` artifact.
- New `tracking` optional dependency (`mlflow>=2.20.0,<3.0.0`) in
  `pyproject.toml`, baked into the training image via `uv sync --extra
  tracking` in the `Dockerfile`.
- Kubernetes wiring: `MLFLOW_TRACKING_URI` in `k8s/base/training-job.yaml`,
  per-overlay `MLFLOW_EXPERIMENT_NAME` in the dev and prod patches, and
  `POD_NAMESPACE` exposed via the downward API for run tagging.

Out of scope (deferred to a later phase):

- Baking a GHCR-published MLflow server image. **Shipped in Phase 1b-2**;
  the platform Deployment now pulls `ghcr.io/taco-ops/nebulift-mlflow`
  instead of installing MLflow at container start. See
  `docs/PHASE_1B_PLATFORM.md`.
- MLflow Model Registry usage. We log the checkpoint as a plain file artifact;
  promoting to a registered model is a Phase 3 (KServe) concern.
- Argo Workflows driving the training Job. Phase 2.
- Argo CD Image Updater. Without it, `IMAGE_TAG` / `COMMIT_SHA` tags on runs
  remain empty unless an overlay hard-codes them.
- Distributed metric reduction. Each rank still trains on its own shard and
  the validation set is evaluated unsharded; rank 0 logs its own metrics
  unchanged.
- Authentication on the MLflow endpoint. Phase 1a's "LAN-trusted" model still
  applies.

## What Gets Logged

Rank 0 of every Job creates exactly one MLflow run inside the configured
experiment.

### Run metadata

| Field           | Source                                             |
|-----------------|----------------------------------------------------|
| Experiment      | `MLFLOW_EXPERIMENT_NAME` env (per-overlay)         |
| Run name        | `MLFLOW_RUN_NAME` env, falling back to `POD_NAME`  |
| Tracking URI    | `MLFLOW_TRACKING_URI` env                          |

### Tags (only set when the env var is non-empty)

| Tag key                  | Env var          |
|--------------------------|------------------|
| `nebulift.environment`   | `ENVIRONMENT`    |
| `nebulift.image_tag`     | `IMAGE_TAG`      |
| `nebulift.commit_sha`    | `COMMIT_SHA`     |
| `nebulift.pod_name`      | `POD_NAME`       |
| `nebulift.pod_namespace` | `POD_NAMESPACE`  |
| `nebulift.job_name`      | `JOB_NAME`       |
| `nebulift.world_size`    | `WORLD_SIZE`     |

### Parameters (logged once at run start)

`epochs`, `batch_size`, `learning_rate`, `world_size`, `num_workers`,
`backend`, `num_classes`, `train_records`, `val_records`,
`model_output_path`. All values are coerced to strings; anything longer than
6000 characters is truncated with a `...[truncated]` marker so it remains
visible in the UI.

### Metrics (per epoch, step = 1-based epoch number)

`train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `learning_rate`.
NaN and infinity values are dropped silently to avoid 400 responses from the
MLflow REST API.

### Artifacts

The final checkpoint (`nebulift_distributed.pth`) is uploaded as
`model/nebulift_distributed.pth` once `_save_unwrapped_checkpoint` returns on
rank 0. The trainer uses MLflow's proxied artifact mode (Phase 1a's
`--serve-artifacts`), so the pod needs no S3 credentials and no boto3
install; the tracking server proxies the upload into MinIO.

## Configuration

All configuration is environment-driven. The defaults shipped in the
manifests assume the Phase 1a in-cluster MLflow Service.

| Env var                  | Default in `k8s/base`                                       | Notes                                  |
|--------------------------|-------------------------------------------------------------|----------------------------------------|
| `MLFLOW_TRACKING_URI`    | `http://mlflow.nebulift-platform.svc.cluster.local:5000`    | Empty → tracking disabled.             |
| `MLFLOW_EXPERIMENT_NAME` | unset in base                                               | Set per overlay (dev/prod).            |
| `MLFLOW_RUN_NAME`        | unset                                                       | Falls back to `POD_NAME`.              |
| `POD_NAME`               | downward API (`metadata.name`)                              | Used as default run name and run tag.  |
| `POD_NAMESPACE`          | downward API (`metadata.namespace`)                         | Run tag only.                          |

The development overlay sets `MLFLOW_EXPERIMENT_NAME=nebulift-development`.
The production overlay sets `MLFLOW_EXPERIMENT_NAME=nebulift-production`. Run
the overlay renderer to confirm:

```bash
kubectl kustomize k8s/overlays/development | grep -A1 MLFLOW
kubectl kustomize k8s/overlays/production  | grep -A1 MLFLOW
```

## Failure Semantics

| Failure                                | Effect on training Job          | Effect on MLflow run            |
|----------------------------------------|---------------------------------|---------------------------------|
| `MLFLOW_TRACKING_URI` unset / empty    | Job succeeds.                   | No run created.                 |
| `mlflow` package not installed         | Job succeeds.                   | No run created; warning logged. |
| Tracking server unreachable            | Job succeeds.                   | No run created; exception logged.|
| Run started, server fails mid-job      | Job succeeds.                   | Run keeps last successful state.|
| Run started, training raises           | Job exits non-zero (existing).  | Run ended with status `FAILED`. |
| `epoch_callback` itself raises         | Job succeeds.                   | Exception logged inside `train()`.|

The single guarantee is: **a tracking-server problem can never fail a
training Job that would otherwise have succeeded.**

## Verification

### From an external workstation

After a Job completes, open `http://mlflow.nebulift.local`, switch to the
experiment matching the overlay (`nebulift-development` or
`nebulift-production`), and inspect the most recent run. Confirm:

- Parameters tab lists the values above.
- Metrics tab shows step-indexed `train_loss`, `train_accuracy`,
  `val_loss`, `val_accuracy`, `learning_rate` series.
- Artifacts tab contains a `model/` directory with
  `nebulift_distributed.pth`.
- Tags include at least `nebulift.environment` and `nebulift.pod_name`.

### From inside the cluster (without UI)

```bash
kubectl run -it --rm mlflow-probe \
  --image=python:3.12-slim \
  --restart=Never \
  -- \
  bash -c '
    pip install --quiet mlflow==2.20.0 &&
    python -c "
import mlflow
mlflow.set_tracking_uri(\"http://mlflow.nebulift-platform.svc.cluster.local:5000\")
client = mlflow.tracking.MlflowClient()
for exp in client.search_experiments():
    print(exp.experiment_id, exp.name)
"
  '
```

### From the training pod logs

The tracker emits a one-line confirmation when it enters:

```text
MLflow tracking enabled: uri=http://... experiment=nebulift-development run_name=nebulift-training-...
```

A disabled tracker logs a single info line stating why
(`set MLFLOW_TRACKING_URI to enable.`), then never touches MLflow again.

## Implementation Notes

- `nebulift.distributed.mlflow_tracker.tracker_from_env(rank=...)` returns a
  no-op tracker for any rank != 0. This keeps the call site identical across
  ranks and avoids creating N duplicate runs in a distributed Job.
- `mlflow` is imported lazily inside `MLflowTracker.__enter__`, not at module
  import time. Local CLI installs without the `tracking` extra continue to
  work; only the container image (built with `uv sync --extra tracking`)
  pulls the dependency.
- The `epoch_callback` parameter on `ModelTrainer.train()` is part of the
  base trainer (not the distributed subclass) so future trainers and tests
  can subscribe to per-epoch metrics without depending on MLflow.
- Callback errors are caught and logged inside `train()` so a buggy observer
  cannot abort a multi-hour job.
- The `MLFLOW_TRACKING_URI` env var lives in `k8s/base/training-job.yaml`,
  not in the per-overlay ConfigMap, because the ConfigMap keys defined under
  `k8s/overlays/*/training-*.conf` are not currently wired into the
  container env. Inline `env:` entries are the existing pattern used by
  `ENVIRONMENT`, `WORLD_SIZE`, and friends; the MLflow vars follow suit.

## Known Limitations and Open Work

- **MLflow server now baked.** Phase 1b-2 swapped the inline-pip Deployment
  for the GHCR `nebulift-mlflow` image. See `docs/PHASE_1B_PLATFORM.md`.
- **No automatic image / commit metadata.** `IMAGE_TAG` and `COMMIT_SHA`
  tags depend on an external tool (Argo CD Image Updater or a CI-driven
  overlay rewrite) and remain empty in the current manifests.
- **No model registry promotion.** Artifacts are logged as plain files; a
  KServe-aware promotion flow is a Phase 3 concern.
- **No distributed metric reduction.** Per-epoch metrics reflect rank 0's
  shard / view of the validation set, not a globally-averaged figure.
- **MLflow has no authentication.** Anything on the LAN that can reach
  `mlflow.nebulift.local` can write to it. This matches Phase 1a's trust
  model.

## Pointers

- Phase 1a platform (prerequisite): `docs/PHASE_1A_PLATFORM.md`
- Trainer entrypoint: `nebulift/distributed/k8s_trainer.py`
- Tracker module: `nebulift/distributed/mlflow_tracker.py`
- Base `ModelTrainer.train()`: `nebulift/ml_model.py`
- Training Job manifests: `k8s/base/training-job.yaml`,
  `k8s/overlays/development/training-job-patch.yaml`,
  `k8s/overlays/production/training-job-patch.yaml`
- Repository overview: `README.md`
