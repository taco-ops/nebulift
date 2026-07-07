#!/usr/bin/env bash
# Seal MinIO root credentials into k8s/platform/base/minio/sealedsecret.yaml.
#
# Usage:
#   scripts/seal-minio-secret.sh
#     -> prompts interactively for accesskey + secretkey
#
#   MINIO_ROOT_USER=admin MINIO_ROOT_PASSWORD=changeme scripts/seal-minio-secret.sh
#     -> non-interactive, takes values from env
#
# Requirements:
#   - kubeseal CLI installed (brew install kubeseal)
#   - kubectl context pointed at the cluster running sealed-secrets controller
#   - sealed-secrets controller installed in kube-system (or override below)
#
# Behavior:
#   - Reads the cluster's sealed-secrets public cert via the controller.
#   - Encrypts (accesskey, secretkey) under namespace=nebulift-platform,
#     name=minio-root (matches the SealedSecret manifest).
#   - Rewrites sealedsecret.yaml in-place with the real encrypted blobs.
#
# Re-run this script any time you rotate the MinIO root credentials. The
# resulting sealedsecret.yaml is safe to commit to git.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEALED_SECRET_PATH="${REPO_ROOT}/k8s/platform/base/minio/sealedsecret.yaml"

SEALED_SECRETS_NAMESPACE="${SEALED_SECRETS_NAMESPACE:-kube-system}"
SEALED_SECRETS_CONTROLLER="${SEALED_SECRETS_CONTROLLER:-sealed-secrets-controller}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' is required but not installed." >&2
    echo "       Install with: brew install $1" >&2
    exit 1
  }
}

require_cmd kubeseal
require_cmd kubectl

if [[ -z "${MINIO_ROOT_USER:-}" ]]; then
  read -r -p "MINIO_ROOT_USER (accesskey): " MINIO_ROOT_USER
fi

if [[ -z "${MINIO_ROOT_PASSWORD:-}" ]]; then
  read -r -s -p "MINIO_ROOT_PASSWORD (secretkey): " MINIO_ROOT_PASSWORD
  echo
fi

if [[ -z "${MINIO_ROOT_USER}" || -z "${MINIO_ROOT_PASSWORD}" ]]; then
  echo "ERROR: MINIO_ROOT_USER and MINIO_ROOT_PASSWORD must be non-empty." >&2
  exit 1
fi

if [[ ${#MINIO_ROOT_PASSWORD} -lt 8 ]]; then
  echo "ERROR: MINIO_ROOT_PASSWORD must be at least 8 characters." >&2
  exit 1
fi

echo "Verifying kubectl context can reach the sealed-secrets controller..."
if ! kubectl -n "${SEALED_SECRETS_NAMESPACE}" get deploy "${SEALED_SECRETS_CONTROLLER}" >/dev/null 2>&1; then
  echo "ERROR: sealed-secrets controller '${SEALED_SECRETS_CONTROLLER}' not found in namespace '${SEALED_SECRETS_NAMESPACE}'." >&2
  echo "       Override with SEALED_SECRETS_NAMESPACE / SEALED_SECRETS_CONTROLLER env vars." >&2
  exit 1
fi

echo "Sealing credentials for namespace=nebulift-platform, name=minio-root..."

TMP_PLAIN="$(mktemp)"
TMP_SEALED="$(mktemp)"
trap 'rm -f "${TMP_PLAIN}" "${TMP_SEALED}"' EXIT

cat >"${TMP_PLAIN}" <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: minio-root
  namespace: nebulift-platform
type: Opaque
stringData:
  accesskey: ${MINIO_ROOT_USER}
  secretkey: ${MINIO_ROOT_PASSWORD}
EOF

kubeseal \
  --controller-namespace="${SEALED_SECRETS_NAMESPACE}" \
  --controller-name="${SEALED_SECRETS_CONTROLLER}" \
  --format=json \
  --scope=strict \
  <"${TMP_PLAIN}" \
  >"${TMP_SEALED}"

# Preserve labels from the existing manifest by merging into a known template.
# kubeseal output is the canonical form; just enrich with our labels.
cat >"${SEALED_SECRET_PATH}" <<'HEADER'
# Sealed MinIO root credentials.
#
# Regenerate with: scripts/seal-minio-secret.sh
#
# Keys consumed by the MinIO Deployment:
#   - accesskey -> $MINIO_ROOT_USER
#   - secretkey -> $MINIO_ROOT_PASSWORD
#
# MLflow also reads the same secret as AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
HEADER

# Strip creationTimestamp if kubeseal emits it, and inject our labels under
# metadata. Use JSON so this script only needs Python's stdlib on workstations.
python3 - "${TMP_SEALED}" "${SEALED_SECRET_PATH}" <<'PYEOF'
import json
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]

with open(src_path) as f:
    doc = json.load(f)

metadata = doc.setdefault("metadata", {})
metadata.pop("creationTimestamp", None)
labels = metadata.setdefault("labels", {})
labels.setdefault("app.kubernetes.io/name", "minio")
labels.setdefault("app.kubernetes.io/component", "object-storage")
labels.setdefault("app.kubernetes.io/part-of", "nebulift")

template = doc.setdefault("spec", {}).setdefault("template", {})
template_meta = template.setdefault("metadata", {})
template_meta["name"] = "minio-root"
template_meta["namespace"] = "nebulift-platform"
template_meta.setdefault("labels", {}).update({
    "app.kubernetes.io/name": "minio",
    "app.kubernetes.io/component": "object-storage",
    "app.kubernetes.io/part-of": "nebulift",
})
template["type"] = "Opaque"

with open(dst_path, "a") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
PYEOF

echo
echo "Sealed secret written to: ${SEALED_SECRET_PATH#"${REPO_ROOT}"/}"
echo "Commit it and let Argo CD sync the nebulift-platform Application."
