# Phase 1a: MLflow + MinIO Platform

## Summary

Phase 1a delivers the MLOps platform that subsequent training, evaluation, and serving phases depend on. It deploys two GitOps-managed services into the new `nebulift-platform` namespace on the homelab k3s cluster:

- **MinIO** — single-replica S3-compatible object store, NFS-backed, exposing both the S3 API and the web console through Traefik Ingress routes.
- **MLflow tracking server** — single-replica, SQLite-on-NFS backend store, configured in `--serve-artifacts` proxy mode so that clients only need the MLflow HTTP endpoint to log and read artifacts.

No trainer code changes ship in Phase 1a. The existing `K8sDistributedTrainer` path continues to checkpoint to NFS as before. MLflow integration in the trainer lands in Phase 1b-1 (see `docs/PHASE_1B_TRAINER.md`).

## Scope and Non-Goals

In scope:

- Cluster-side platform manifests under `k8s/platform/base/`.
- Argo CD `Application` and the AppProject changes required to manage Ingress + SealedSecret resources.
- One-time cluster bootstrap checks for Traefik Ingress and Sealed Secrets under `cluster-bootstrap/`.
- A `kubeseal` helper script for rotating the MinIO root credential.

Out of scope (Phase 1b or later):

- Any change to the training image or `nebulift.distributed.k8s_trainer` entrypoint.
- Authentication or TLS termination at Ingress. The platform listens on HTTP/80 only and trusts the LAN. Cert-manager + HTTPS is a separate phase.
- Backups, HA, multi-tenant access policies, or quota enforcement on MinIO.
- A custom GHCR-published MLflow image. Phase 1a installed MLflow via inline `pip install` in `python:3.11-slim`; the baked image shipped in Phase 1b-2. See `docs/PHASE_1B_PLATFORM.md`.

## Components

### Cluster bootstrap (manual, one-time)

- `cluster-bootstrap/README.md` — commands to verify the Sealed Secrets controller and bundled Traefik Ingress controller before the platform Application syncs.
- `cluster-bootstrap/gateway-api/` and `cluster-bootstrap/traefik/helmchartconfig.yaml` — optional Gateway API bootstrap resources retained for future phases; they are not required by the current Ingress-based platform.

These are intentionally kept out of GitOps because the nebulift AppProject is not allowed to write into `kube-system`.

### Platform manifests (`k8s/platform/base/`)

- `namespace.yaml` — `nebulift-platform` namespace with Pod Security Admission labels (`enforce: baseline`, `warn: restricted`).
- `minio/` — Deployment (sync-wave `"1"`, `Recreate`, ARM64 affinity, NFS volume), Service (ports `s3:9000`, `console:9001`), SealedSecret (`minio-root` with keys `accesskey`/`secretkey`), two Ingresses (`minio.nebulift.local` → console, `s3.nebulift.local` → S3 API), and a bucket-bootstrap Job (sync-wave `"2"`, `argocd.argoproj.io/hook: Sync`, `hook-delete-policy: BeforeHookCreation`) that creates the `mlflow-artifacts` bucket.
- `mlflow/` — ConfigMap (sqlite URI, S3 endpoint), Service (`http:5000`), Deployment (sync-wave `"3"`, ARM64 affinity, NFS volume for the SQLite database, inline `pip install mlflow==2.20.0 boto3` on container start as originally shipped; replaced by the baked GHCR image in Phase 1b-2), and an Ingress (`mlflow.nebulift.local`).
- `kustomization.yaml` — top-level base that aggregates all of the above. Uses the modern `labels` field with `includeSelectors: false` so the `app.kubernetes.io/part-of=nebulift` and `app.kubernetes.io/managed-by=argocd` labels are applied to every resource without polluting Deployment selectors. The `nebulift.io/phase: "1a"` common annotation marks every resource for easy identification.

### GitOps resources

- `argocd/applications/nebulift-platform.yaml` — standalone Argo CD Application that tracks `main`, syncs `k8s/platform/base/` into `nebulift-platform`, and uses `ServerSideApply=true` to keep large SealedSecret annotations safe. Sync-wave `"0"` is retained so future workload orchestrators can depend on the platform being present first.
- `argocd/projects/nebulift-project.yaml` — AppProject extended with `networking.k8s.io/Ingress` and `bitnami.com/SealedSecret` in `namespaceResourceWhitelist`.

### Tooling

- `scripts/seal-minio-secret.sh` — wraps `kubeseal` and writes the sealed payload into `k8s/platform/base/minio/sealedsecret.yaml`. Reads `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` from the environment or prompts interactively. Enforces a minimum password length of 8. Overridable via `SEALED_SECRETS_NAMESPACE` (default `kube-system`) and `SEALED_SECRETS_CONTROLLER` (default `sealed-secrets-controller`).

## Architecture

```text
                          LAN (10.42.0.0/24)
                                 |
                  *.nebulift.local resolves to k3s ingress IP
                                 |
                        +--------v---------+
                        |  Traefik (k3s)   |
                        |  Ingress         |
                        +--------+---------+
                                 |
                  +--------------+---------------+
                  |              |               |
         mlflow.nebulift.local   minio...   s3.nebulift.local
                  |              |               |
              +---v---+      +---v----+      +---v----+
              | MLflow|      | MinIO  |      | MinIO  |
              | :5000 |      | :9001  |      | :9000  |
              +---+---+      +---+----+      +---+----+
                  |              |               |
                  | proxy reads/writes via S3 endpoint inside cluster
                  +------------> minio.nebulift-platform.svc:9000
                  |
                  | --serve-artifacts -> boto3 -> mlflow-artifacts bucket
                  |
              +---v------------------------+
              | sqlite:////mlflow/db/...   |
              |   NFS 10.42.0.22 :          |
              |   /exports/cluster-storage/ |
              |     nebulift-platform/      |
              |       mlflow/db/            |
              +-----------------------------+

   MinIO data path: NFS 10.42.0.22:/exports/cluster-storage/nebulift-platform/minio
```

Sync-wave ordering inside the platform Application:

```text
wave 0  namespace + sealedsecret + mlflow ConfigMap
wave 1  MinIO Deployment + Service
wave 2  minio-bucket-bootstrap Job (argocd Sync hook)
wave 3  MLflow Deployment + Service + Ingress
```

## Initial Bootstrap

Run these in order. The first checks are one-time per cluster.

### 1. Verify bootstrap components

See `cluster-bootstrap/README.md` for the verified commands. Verify with:

```bash
kubectl get deploy -n kube-system sealed-secrets-controller
kubectl get ingressclass traefik
```

If Sealed Secrets is missing, install the official controller from
`bitnami-labs/sealed-secrets` releases. The platform manifests assume it lives
at `kube-system/sealed-secrets-controller`.

### 2. Seal the MinIO root credential

The committed `k8s/platform/base/minio/sealedsecret.yaml` must be sealed for
the target cluster. Generate or rotate the sealed payload before first sync:

```bash
export MINIO_ROOT_USER="nebulift-admin"
export MINIO_ROOT_PASSWORD="<at-least-8-chars>"
./scripts/seal-minio-secret.sh
```

The script overwrites `k8s/platform/base/minio/sealedsecret.yaml` in place. Commit the regenerated file before pointing Argo CD at the branch. The plaintext credential never leaves your shell.

### 3. Apply the AppProject changes and the Application

```bash
kubectl apply -f argocd/projects/nebulift-project.yaml
kubectl apply -f argocd/applications/nebulift-platform.yaml
```

Argo CD will then create the `nebulift-platform` namespace and roll out the platform in the wave order above.

### 4. Configure LAN DNS

Point the following hostnames at the k3s Traefik ingress IP (the same IP that already serves any pre-existing k3s `Ingress`):

- `mlflow.nebulift.local`
- `minio.nebulift.local`
- `s3.nebulift.local`

Use your router's DNS, `/etc/hosts`, or your local DNS server (e.g. Pi-hole, dnsmasq). No TLS is configured; these are HTTP-only.

## Verification

After Argo CD reports `Synced` and `Healthy`:

```bash
kubectl get all -n nebulift-platform
kubectl get ingress -n nebulift-platform
kubectl get sealedsecret,secret -n nebulift-platform

# MinIO bucket exists
kubectl logs -n nebulift-platform job/minio-bucket-bootstrap

# HTTP reachability from your workstation
curl -sI http://mlflow.nebulift.local | head -1
curl -sI http://minio.nebulift.local | head -1
curl -sI http://s3.nebulift.local/minio/health/ready | head -1
```

The MLflow UI should load at `http://mlflow.nebulift.local`. The MinIO console at `http://minio.nebulift.local`. The S3 API responds at `http://s3.nebulift.local`.

## Rotating the MinIO Credential

```bash
export MINIO_ROOT_USER="nebulift-admin"
export MINIO_ROOT_PASSWORD="<new-password>"
./scripts/seal-minio-secret.sh
git add k8s/platform/base/minio/sealedsecret.yaml
git commit -m "chore(platform): rotate MinIO root credential"
git push
```

Argo CD will roll the SealedSecret. Because the MinIO Deployment uses `Recreate` strategy and the env vars are read on container start, you must restart the pod for the new credential to take effect:

```bash
kubectl rollout restart deployment/minio -n nebulift-platform
```

The `minio-bucket-bootstrap` Job re-runs on the next sync because of the `argocd.argoproj.io/hook: Sync` annotation and will validate that the new credential works.

## Known Limitations and Open Work

- **MinIO on NFS is suboptimal.** NFS does not provide the POSIX semantics MinIO recommends. For a homelab this is acceptable, but expect slower performance and the small risk of corruption under concurrent writes. If problems appear, the documented fallback is `hostPath` + `nodeAffinity` pinning MinIO to a single node.
- **SQLite-on-NFS is single-writer.** The MLflow Deployment uses `replicas: 1` and `strategy: Recreate` to enforce this. Do not scale MLflow up.
- **Inline `pip install` on container start** was the original Phase 1a deployment shape; it added ~30s of cold-start latency and made the platform PyPI-dependent. Phase 1b-2 replaced it with a baked GHCR image (`ghcr.io/taco-ops/nebulift-mlflow`); see `docs/PHASE_1B_PLATFORM.md`.
- **No HTTPS.** The Ingress routes listen on HTTP/80 only. Cert-manager + HTTPS is a separate phase.
- **No authentication on MLflow.** MLflow has no built-in authentication. Anyone on the LAN who can reach `mlflow.nebulift.local` can write to it. This matches the trust model of the homelab; it must be revisited before any external exposure.
- **Backups.** NFS snapshots are the only safety net. Add `restic` or similar before relying on MLflow run history.
- **Trainer integration shipped in Phase 1b-1.** `K8sDistributedTrainer` now logs params, per-epoch metrics, and the final checkpoint to MLflow from rank 0; see `docs/PHASE_1B_TRAINER.md`. The MLflow server itself was reshipped as a baked GHCR image in Phase 1b-2; see `docs/PHASE_1B_PLATFORM.md`.

## Pointers

- One-time cluster bootstrap: `cluster-bootstrap/README.md`
- Sealed-secret rotation helper: `scripts/seal-minio-secret.sh`
- Argo CD Application: `argocd/applications/nebulift-platform.yaml`
- Argo CD AppProject: `argocd/projects/nebulift-project.yaml`
- Repository overview and CLI usage: `README.md`
