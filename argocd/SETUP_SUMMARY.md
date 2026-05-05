# ArgoCD Setup Summary

## What We Created

### ArgoCD Resources
```
argocd/
├── QUICKSTART.md                           # Quick start guide
├── README.md                               # Complete documentation
├── applicationset.yaml                     # Multi-environment management
├── namespaces.yaml                         # Namespace definitions
├── applications/
│   ├── nebulift-training.yaml             # Production app
│   └── nebulift-training-dev.yaml         # Development app
└── projects/
    └── nebulift-project.yaml              # RBAC and permissions
```

### Kustomize Structure
```
k8s/
├── base/                                   # Common resources
│   ├── kustomization.yaml
│   ├── configmap.yaml
│   ├── service.yaml
│   ├── training.conf
│   └── training-job.yaml
└── overlays/
    ├── development/                        # Dev environment
    │   ├── kustomization.yaml
    │   ├── namespace.yaml
    │   ├── training-dev.conf
    │   └── training-job-patch.yaml
    └── production/                         # Prod environment
        ├── kustomization.yaml
        ├── namespace.yaml
        ├── training-prod.conf
        ├── training-job-patch.yaml
        └── configmap-patch.yaml
```

## How to Deploy

### Initial Setup (One Time)

Since you already have ArgoCD running at `argocd.spacedojo.local`:

```bash
# 1. Login
argocd login argocd.spacedojo.local --insecure

# 2. Apply ArgoCD resources
kubectl apply -f argocd/projects/nebulift-project.yaml
kubectl apply -f argocd/applicationset.yaml

# 3. Verify
argocd app list | grep nebulift
```

This creates two applications:
- `nebulift-development` → watches `develop` branch → deploys to `nebulift-dev` namespace
- `nebulift-production` → watches `main` branch → deploys to `nebulift-training` namespace

## GitOps Workflow

### For Development
```bash
# Make changes
vim k8s/overlays/development/training-dev.conf

# Commit and push to develop branch
git checkout develop
git add .
git commit -m "Update dev config"
git push origin develop

# ArgoCD auto-syncs to nebulift-dev namespace
```

### For Production
```bash
# Make changes
vim k8s/overlays/production/training-prod.conf

# Commit and push to main branch
git checkout main
git add .
git commit -m "Update prod config"
git push origin main

# ArgoCD auto-syncs to nebulift-training namespace
```

## Key Features

### ✅ Auto-Sync Enabled
- Changes deploy automatically within ~3 minutes
- Can configure webhooks for instant deployment

### ✅ Self-Healing Enabled
- ArgoCD reverts manual kubectl changes
- Ensures cluster matches Git state

### ✅ Pruning Enabled
- Removes resources deleted from Git
- Keeps cluster clean

### ✅ Environment Isolation
- Dev and prod completely separated
- Different resource limits per environment

### ✅ RBAC Configured
Three roles defined:
- **Admin**: Full control (groups: `nebulift-admins`, `platform-team`)
- **Developer**: Sync and view (groups: `nebulift-developers`, `ml-engineers`)
- **Viewer**: Read-only (groups: `nebulift-viewers`, `data-scientists`)

## Configuration Differences

### Development Environment
- **Branch**: `develop`
- **Namespace**: `nebulift-dev`
- **Pods**: 2 replicas
- **Resources**: 1Gi RAM, 1 CPU
- **Epochs**: 10
- **Batch Size**: 16
- **Log Level**: DEBUG

### Production Environment
- **Branch**: `main`
- **Namespace**: `nebulift-training`
- **Pods**: 8 replicas
- **Resources**: 4Gi RAM, 3 CPU
- **Epochs**: 100
- **Batch Size**: 64
- **Log Level**: INFO

## Common Operations

### View Application Status
```bash
argocd app get nebulift-production
argocd app get nebulift-development
```

### Force Sync (immediate deployment)
```bash
argocd app sync nebulift-production
```

### View Diff (what will be deployed)
```bash
argocd app diff nebulift-production
```

### Rollback
```bash
argocd app history nebulift-production
argocd app rollback nebulift-production
```

### Pause Auto-Sync
```bash
argocd app set nebulift-production --sync-policy none
# ... do manual testing ...
argocd app set nebulift-production --sync-policy automated
```

## Next Steps

1. **Configure Git webhooks** for instant sync (optional)
2. **Set up SSO** for ArgoCD UI access
3. **Configure notifications** (Slack, email) for deployment events
4. **Add more environments** (staging, QA) if needed
5. **Integrate with CI/CD** to update image tags automatically

## Resources

- **Quick Start**: `argocd/QUICKSTART.md`
- **Full Documentation**: `argocd/README.md`
- **ArgoCD UI**: `https://argocd.spacedojo.local`
- **ArgoCD Docs**: https://argo-cd.readthedocs.io/