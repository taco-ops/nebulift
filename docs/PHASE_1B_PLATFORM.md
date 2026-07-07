# Phase 1b-2: Baked MLflow Platform Image

## Summary

Phase 1b-2 replaces the Phase 1a runtime `pip install mlflow boto3` shim with
a custom GHCR-published MLflow server image. The MLflow Deployment in
`k8s/platform/base/mlflow/` now pulls
`ghcr.io/taco-ops/nebulift-mlflow:latest`, an arm64+amd64 multi-arch image
built by CircleCI from `docker/mlflow/Dockerfile`. No trainer-side changes
ship in this phase; the Phase 1b-1 client integration continues to work
unchanged because the tracking endpoint, port, ConfigMap envs, and on-disk
SQLite layout are all preserved.

The migration is in-place: the existing MLflow Service, Ingress, NFS
volume, and SealedSecret are untouched. Argo CD synchronizes the new image
reference into the cluster and Kubernetes performs a single `Recreate` roll
of the one-replica Deployment.

## Scope and Non-Goals

In scope:

- New `docker/mlflow/Dockerfile` baking `mlflow==2.20.0` + `boto3>=1.34,<2`
  on top of `python:3.11-slim-bookworm` with a non-root UID 1000 user and an
  `ENTRYPOINT ["mlflow"]` so the Deployment passes a clean `args:` list.
- New `image_build_mlflow` CircleCI job: multi-arch (`linux/arm64`,
  `linux/amd64`) buildx push to `ghcr.io/taco-ops/nebulift-mlflow` with
  `<short-sha>`, `<branch>`, and (on `main`) `latest` tags; registry-side
  buildcache at `:buildcache`. Gated on `lint_and_typecheck` only, not on
  the trainer test suite, because the platform image shares no source with
  the Python package.
- `k8s/platform/base/mlflow/deployment.yaml` rewrite:
  - `image: ghcr.io/taco-ops/nebulift-mlflow:latest` + `imagePullPolicy: Always`.
  - `command:` removed; `args:` is now a clean MLflow CLI invocation with
    `$(VAR)` substitution from the existing ConfigMap.
  - `HOME=/tmp` env removed (no more user-space pip install).
  - `securityContext.readOnlyRootFilesystem: true`.
  - Readiness / liveness `initialDelaySeconds` reduced (45 → 15, 90 → 30) to
    reflect baked-image startup speed.
- `k8s/platform/base/mlflow/configmap.yaml`: `MLFLOW_VERSION` key dropped;
  the version is now baked into the image and bumped via the Dockerfile.
- `k8s/platform/base/kustomization.yaml`: `nebulift.io/phase` annotation
  bumped to `"1b-2"` to track the current platform revision.

Out of scope (Phase 2 or later):

- Argo CD Image Updater. Without it, `:latest` is the only refresh signal; a
  manual `kubectl rollout restart` is the only way to pin to a known good
  digest after a regression. See "Rollback" below for the workaround.
- Switching the trainer image to a baked GHCR tag for the training Job. The
  Job still uses `imagePullPolicy: IfNotPresent` and is out of scope for
  this phase.
- MLflow authentication, HTTPS, multi-replica MLflow, or PostgreSQL backend.
  All deferred (Phase 1a's trust model still applies).
- Bumping `mlflow` past 2.20.0. The pin is coupled to the trainer client
  floor in `pyproject.toml`; see "Version Coupling" below.

## Image Layout

`docker/mlflow/Dockerfile` is intentionally minimal:

```text
python:3.11-slim-bookworm
  ├── ARG MLFLOW_VERSION=2.20.0
  ├── ARG BOTO3_SPEC="boto3>=1.34,<2"
  ├── user/group mlflow:mlflow (uid 1000, gid 1000), HOME=/home/mlflow
  ├── pip install --no-cache-dir mlflow==${MLFLOW_VERSION} ${BOTO3_SPEC}
  ├── USER mlflow:mlflow, WORKDIR /home/mlflow, EXPOSE 5000
  ├── HEALTHCHECK CMD python -c "...:5000/health..."
  └── ENTRYPOINT ["mlflow"], CMD ["--help"]
```

Why not the upstream `ghcr.io/mlflow/mlflow` image? It does not bundle
`boto3`, so MinIO-backed `--serve-artifacts` would require an init container
or a runtime install layered on top. Building our own image is cheaper than
maintaining a second container, and lets us pin both packages
deterministically.

Why `python:3.11` (not 3.12)? It matches what Phase 1a ran in production
and avoids implicit Python upgrades inside a single PR. The trainer client
side runs on its own Python (3.12) inside the trainer image; the two paths
are decoupled.

Why `boto3>=1.34,<2` instead of an exact pin? boto3 ships frequent patch
releases for new AWS regions and security fixes, and MLflow only uses its
S3 surface. Pinning to a wide compatible range lets `image_build_mlflow`
rebuilds pick up security updates without coordinated bumps.

## CI Pipeline

The new job in `.circleci/config.yml`:

```text
image_build_mlflow
  ├── machine: ubuntu-2204
  ├── starts the Docker daemon and QEMU binfmt handlers
  ├── creates a dedicated `nebulift-mlflow-builder` buildx instance
  ├── logs in to ghcr.io using $GHCR_TOKEN from the `ghcr` context
  └── docker buildx build --platform linux/arm64,linux/amd64
        --tag ghcr.io/<owner>/<repo>-mlflow:<sha>
        --tag ghcr.io/<owner>/<repo>-mlflow:<branch>
        [--tag ghcr.io/<owner>/<repo>-mlflow:latest]  # main only
        --cache-from/--cache-to type=registry,...:buildcache
        --push docker/mlflow/
```

Tags follow the same convention as the trainer image: every push gets a
short-SHA and a branch-name tag; pushes to `main` additionally retag
`:latest`. The buildcache lives at `:buildcache` in the same GHCR package,
so even a cold runner only pays the multi-arch base-image cost on the
first build.

## Deployment Behavior

In-cluster behavior changes:

- **Cold start drops from ~30s to ~5s.** The pod no longer downloads
  `mlflow`, `boto3`, and their dependency tree from PyPI; the image already
  contains them.
- **No PyPI runtime dependency.** A PyPI outage no longer prevents an MLflow
  pod from restarting. The image is served from GHCR (already required for
  the trainer).
- **Filesystem is read-only at runtime.** `readOnlyRootFilesystem: true` is
  now safe because no package install happens in the running container. The
  only writable paths are `/mlflow/db` (NFS) and `/tmp` (emptyDir, 1Gi cap).
- **Pull on every Recreate.** `imagePullPolicy: Always` means an Argo CD
  sync that bumps the manifest, or a `kubectl rollout restart deployment/mlflow`,
  forces kubelet to re-resolve the `:latest` tag and pick up the most recent
  digest pushed by CI. This is intentional: with no Image Updater, the
  rollout-restart is the canonical "deploy the latest MLflow build" command.

Behavior **unchanged**:

- Service name and port (`mlflow.nebulift-platform.svc.cluster.local:5000`).
- Ingress (`mlflow.nebulift.local`).
- SQLite metadata DB on NFS at `/mlflow/db`.
- MinIO-backed artifact storage via `--serve-artifacts`.
- MinIO root credentials sourced from the `minio-root` SealedSecret.

## Verification

After the platform Application syncs:

```bash
# New image reference rolled out.
kubectl get deploy -n nebulift-platform mlflow -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
# Expected: ghcr.io/taco-ops/nebulift-mlflow:latest

# Pod is running on a baked image (no pip install logs).
kubectl logs -n nebulift-platform deploy/mlflow | head -20
# Expected: gunicorn boot output without "Installing collected packages: ..."

# Root filesystem is read-only.
kubectl exec -n nebulift-platform deploy/mlflow -- \
  sh -c 'touch /should-fail 2>&1 || echo "RO confirmed"'
# Expected: "Read-only file system" then "RO confirmed"

# Service still answers /health.
kubectl exec -n nebulift-platform deploy/mlflow -- \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/health').read().decode())"
# Expected: OK

# End-to-end from a Phase 1b-1 trainer Job (development overlay).
kubectl apply -k k8s/overlays/development
kubectl logs -n nebulift -l app.kubernetes.io/name=nebulift-trainer --tail=200 | grep -i mlflow
```

The MLflow UI at `http://mlflow.nebulift.local` should continue to list
existing experiments and runs from the Phase 1a SQLite database (the
metadata path on NFS is unchanged).

## Rollback

If the new image regresses MLflow behavior, the fastest revert is to
re-point the Deployment at the inline-pip Phase 1a shim. Two options:

1. **Revert the Phase 1b-2 manifest commit** (`feat/phase-1b-platform-mlflow-image`).
   Argo CD reapplies the previous `python:3.11-slim` + bash wrapper. This
   is the canonical rollback path.
2. **Pin to a known-good digest** if `:latest` itself is the problem. List
   tags with `gh api /users/taco-ops/packages/container/nebulift-mlflow/versions`
   (requires PAT with `read:packages`), then patch the live Deployment:

   ```bash
   kubectl patch deploy/mlflow -n nebulift-platform \
     --type=json \
     -p='[{"op":"replace","path":"/spec/template/spec/containers/0/image","value":"ghcr.io/taco-ops/nebulift-mlflow:<short-sha>"}]'
   ```

   This is a Git-of-band override and Argo CD will revert it on next sync;
   pair it with a corresponding manifest commit when iterating.

## Version Coupling

The baked MLflow version is coupled to the trainer client floor:

| Component | Pin                                                | Location                       |
|-----------|----------------------------------------------------|--------------------------------|
| Server    | `mlflow==2.20.0` (ARG MLFLOW_VERSION)              | `docker/mlflow/Dockerfile`     |
| Client    | `mlflow>=2.20.0,<3.0.0` (tracking optional extra)  | `pyproject.toml`               |

Bumping the server requires:

1. Update `ARG MLFLOW_VERSION` in `docker/mlflow/Dockerfile`.
2. Bump the floor in `pyproject.toml` if a server feature requires a newer
   client API. Regenerate `uv.lock` (`uv lock`).
3. Run the trainer test suite (`uv run pytest tests/test_mlflow_tracker.py
   tests/test_ml_model.py -v`) to validate the client side still parses
   server responses.
4. Push to a feature branch; CircleCI builds and pushes the new image. Argo
   CD picks up `:latest` on next sync, or pin to the new `<sha>` for a
   controlled rollout.
5. If MLflow 3.x is on the table, the `<3.0.0` ceiling in `pyproject.toml`
   must be lifted in the same PR; expect breaking server/client changes.

Never bump only one side.

## Known Limitations and Open Work

- **No Image Updater.** `:latest` floats forward on every push to `main`,
  but a running pod only refreshes on `Recreate`. Document this in the
  ops runbook; or adopt Argo CD Image Updater in a future phase.
- **`COMMIT_SHA` / `IMAGE_TAG` are still empty on training runs.** The
  trainer manifest does not inject the running image's tag into pod env.
  Inherited from Phase 1b-1; not addressed here.
- **No PostgreSQL backend.** SQLite on NFS remains the metadata store with
  the same single-writer caveat as Phase 1a.
- **No HTTPS, no MLflow auth.** Unchanged from Phase 1a.
- **Trainer image not converted.** `k8s/base/training-job.yaml` still uses
  `imagePullPolicy: IfNotPresent`; a `:latest` push there would not refresh
  on existing nodes without a manual evict / restart.

## Pointers

- Image source: `docker/mlflow/Dockerfile`
- CI job: `.circleci/config.yml` → `jobs.image_build_mlflow`
- Deployment: `k8s/platform/base/mlflow/deployment.yaml`
- ConfigMap: `k8s/platform/base/mlflow/configmap.yaml`
- Platform kustomize root: `k8s/platform/base/kustomization.yaml`
- Phase 1a platform (predecessor): `docs/PHASE_1A_PLATFORM.md`
- Phase 1b-1 trainer integration: `docs/PHASE_1B_TRAINER.md`
- Repository overview: `README.md`
