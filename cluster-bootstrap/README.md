# Cluster bootstrap

One-time, cluster-scoped resources that must exist **before** Argo CD syncs the
nebulift-platform Application. These are intentionally NOT managed by Argo CD
because:

- The `nebulift` AppProject restricts destinations to `nebulift-*` namespaces +
  `argocd`. CRDs and `kube-system` resources cannot be deployed under it.
- The Sealed Secrets controller owns a cluster keypair used to decrypt committed
  `SealedSecret` resources into native `Secret` resources. It must exist before
  the platform Application syncs the MinIO credential.
- Optional Gateway API CRDs and Traefik provider overrides are infrastructure
  bootstrap, not application lifecycle. They change rarely and should be applied
  deliberately if a future phase reintroduces Gateway API resources.

## What's here

| Path | Purpose | Scope |
|------|---------|-------|
| `gateway-api/` | Optional Gateway API v1 standard-channel CRDs (`GatewayClass`, `Gateway`, `HTTPRoute`, `ReferenceGrant`) for future Gateway API work. | Cluster |
| `traefik/helmchartconfig.yaml` | Optional k3s Traefik Gateway API provider override. Not required for the current Ingress-based platform. | `kube-system` |

## Initial setup (run once per cluster)

```bash
# 1. Verify the Sealed Secrets controller if the cluster already has it.
kubectl get crd sealedsecrets.bitnami.com
kubectl get deploy -n kube-system sealed-secrets-controller

# 2. Verify Traefik Ingress is available.
kubectl get ingressclass traefik
```

Before the first platform sync, generate or validate a real MinIO
`SealedSecret` for this cluster:

```bash
scripts/seal-minio-secret.sh
```

The committed ciphertext in `k8s/platform/base/minio/sealedsecret.yaml` is
cluster-specific. Validate it against the current controller with:

```bash
kubeseal --validate \
  --controller-namespace=kube-system \
  --controller-name=sealed-secrets-controller \
  < k8s/platform/base/minio/sealedsecret.yaml
```

## Optional Gateway API Setup

The current platform uses standard Kubernetes `Ingress` resources. If a future
phase reintroduces Gateway API resources, install the CRDs and enable the k3s
Traefik Gateway API provider deliberately:

```bash
kubectl apply -k cluster-bootstrap/gateway-api
kubectl apply -f cluster-bootstrap/traefik/helmchartconfig.yaml
kubectl -n kube-system rollout restart deploy/traefik
kubectl get gatewayclass traefik -o jsonpath='{.status.conditions[?(@.type=="Accepted")].status}'
```

Validate Gateway API upgrades against the release notes for breaking schema
changes:
https://github.com/kubernetes-sigs/gateway-api/releases
