# Cluster bootstrap

One-time, cluster-scoped resources that must exist **before** Argo CD syncs the
nebulift-platform Application. These are intentionally NOT managed by Argo CD
because:

- The `nebulift` AppProject restricts destinations to `nebulift-*` namespaces +
  `argocd`. CRDs and `kube-system` resources cannot be deployed under it.
- Gateway API CRDs and the Traefik provider override are infrastructure
  bootstrap, not application lifecycle. They change rarely and should be applied
  deliberately.

## What's here

| Path | Purpose | Scope |
|------|---------|-------|
| `gateway-api/` | Gateway API v1 standard-channel CRDs (`GatewayClass`, `Gateway`, `HTTPRoute`, `ReferenceGrant`). | Cluster |
| `traefik/helmchartconfig.yaml` | Enables Traefik's Gateway API provider in k3s. | `kube-system` |

## Initial setup (run once per cluster)

```bash
# 1. Install Gateway API CRDs.
kubectl apply -k cluster-bootstrap/gateway-api

# 2. Enable Traefik's Gateway API provider.
kubectl apply -f cluster-bootstrap/traefik/helmchartconfig.yaml
kubectl -n kube-system rollout restart deploy/traefik

# 3. Verify the auto-created GatewayClass is Accepted.
kubectl get gatewayclass traefik -o jsonpath='{.status.conditions[?(@.type=="Accepted")].status}'
# Expect: True
```

After this, the Argo CD `nebulift-platform` Application can create a `Gateway`
(referencing `gatewayClassName: traefik`) and `HTTPRoute`s under the
`nebulift-platform` namespace.

## Upgrading Gateway API

```bash
# Edit cluster-bootstrap/gateway-api/kustomization.yaml to bump the version,
# then:
kubectl apply -k cluster-bootstrap/gateway-api
kubectl -n kube-system rollout restart deploy/traefik
```

Validate each upgrade against the Gateway API release notes for breaking
schema changes:
https://github.com/kubernetes-sigs/gateway-api/releases
