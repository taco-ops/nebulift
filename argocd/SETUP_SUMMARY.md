# Argo CD Setup Summary

## Resources Added

```text
argocd/
├── QUICKSTART.md
├── README.md
├── applicationset.yaml
├── namespaces.yaml
├── applications/
│   ├── nebulift-training.yaml
│   └── nebulift-training-dev.yaml
└── projects/
    └── nebulift-project.yaml
```

```text
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── configmap.yaml
│   ├── service.yaml
│   ├── training.conf
│   └── training-job.yaml
└── overlays/
    ├── development/
    │   ├── kustomization.yaml
    │   ├── namespace.yaml
    │   ├── training-dev.conf
    │   └── training-job-patch.yaml
    └── production/
        ├── kustomization.yaml
        ├── namespace.yaml
        ├── training-prod.conf
        ├── training-job-patch.yaml
        └── configmap-patch.yaml
```

## Initial Setup

If Argo CD is available at `argocd.spacedojo.local`, run:

```bash
argocd login argocd.spacedojo.local --insecure
kubectl apply -f argocd/projects/nebulift-project.yaml
kubectl apply -f argocd/applicationset.yaml
argocd app list | grep nebulift
```

This creates:

- `nebulift-development`: watches `develop` and deploys to `nebulift-dev`
- `nebulift-production`: watches `main` and deploys to `nebulift-training`

## GitOps Workflow

Development changes:

```bash
git checkout develop
vim k8s/overlays/development/training-dev.conf
git add k8s/overlays/development/training-dev.conf
git commit -m "Update development training config"
git push origin develop
```

Production changes:

```bash
git checkout main
vim k8s/overlays/production/training-prod.conf
git add k8s/overlays/production/training-prod.conf
git commit -m "Update production training config"
git push origin main
```

## Enabled Behaviors

- Auto-sync reconciles Git changes into the target namespace.
- Self-healing returns manually changed resources to the Git-defined state.
- Pruning removes resources that were deleted from Git.
- Development and production use separate namespaces and overlays.
- The AppProject defines admin, developer, and viewer roles.

## Environment Configuration

Development:

- Branch: `develop`
- Namespace: `nebulift-dev`
- Pods: 2 replicas
- Resources: 1 GiB RAM, 1 CPU
- Epochs: 10
- Batch size: 16
- Log level: DEBUG

Production:

- Branch: `main`
- Namespace: `nebulift-training`
- Pods: 8 replicas
- Resources: 4 GiB RAM, 3 CPU
- Epochs: 100
- Batch size: 64
- Log level: INFO

## Common Operations

```bash
argocd app get nebulift-production
argocd app get nebulift-development
argocd app sync nebulift-production
argocd app diff nebulift-production
argocd app history nebulift-production
argocd app rollback nebulift-production
```

Temporarily pause auto-sync:

```bash
argocd app set nebulift-production --sync-policy none
argocd app set nebulift-production --sync-policy automated
```

## Next Steps

1. Configure Git webhooks if faster sync is required.
2. Configure SSO for Argo CD UI access.
3. Configure notifications for failed syncs or degraded applications.
4. Add staging or QA overlays if needed.
5. Add CI render checks before relying on automated production sync.

## References

- Quick start: `argocd/QUICKSTART.md`
- Full documentation: `argocd/README.md`
- Argo CD UI: `https://argocd.spacedojo.local`
- Argo CD docs: https://argo-cd.readthedocs.io/
