# Nebulift ArgoCD Deployment Guide

This guide explains how to deploy Nebulift using ArgoCD for GitOps-based continuous deployment.

## What is GitOps?

**GitOps** is a deployment methodology where:
- **Git is the single source of truth** for infrastructure and application state
- **ArgoCD continuously monitors** the Git repository for changes
- **Changes are automatically deployed** when you push to Git
- **Manual kubectl deployment commands are avoided** for managed resources
- **Drift detection and self-healing** keeps cluster state synchronized with Git

### The GitOps Workflow

```
Developer → Git Commit → Git Push → ArgoCD Detects → ArgoCD Syncs → Kubernetes Cluster
```

**Key Principle**: You never manually apply Kubernetes manifests. ArgoCD does it automatically based on what's in Git.

## Architecture Overview

The ArgoCD deployment consists of:

- **ApplicationSet**: Manages multiple environments (dev, production)
- **Applications**: Individual application instances per environment
- **AppProject**: RBAC and resource management for Nebulift
- **Kustomize Overlays**: Environment-specific configurations

**Git Branches → Environments Mapping**:
- `develop` branch → Development environment (`nebulift-dev` namespace)
- `main` branch → Production environment (`nebulift-training` namespace)

## Prerequisites

### 1. ArgoCD Installation
Ensure ArgoCD is installed and accessible in your cluster:

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### 2. Repository Access
Configure ArgoCD to access the Nebulift repository:

```bash
# Add repository to ArgoCD
argocd repo add https://github.com/taco-ops/nebulift.git --username <username> --password <token>
```

### 3. Container Registry Access
Ensure your cluster can pull from `ghcr.io/taco-ops/nebulift`:

```bash
# Create image pull secret if needed
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<token> \
  --namespace=nebulift-training
```

## Quick Start Deployment

### Step 1: Login to ArgoCD

```bash
# Login to your ArgoCD instance
argocd login argocd.spacedojo.local --insecure

# Or access via port-forward if needed
kubectl port-forward svc/argocd-server -n argocd 8080:443
argocd login localhost:8080 --insecure
```

### Step 2: Deploy ArgoCD Resources

**One-time setup**: Apply the ArgoCD project and application configurations to your cluster:

```bash
# Deploy the Nebulift project (RBAC and permissions)
kubectl apply -f argocd/projects/nebulift-project.yaml

# Option A: Deploy ApplicationSet (manages both dev and prod)
kubectl apply -f argocd/applicationset.yaml

# Option B: Deploy individual applications
kubectl apply -f argocd/applications/nebulift-training-dev.yaml
kubectl apply -f argocd/applications/nebulift-training.yaml
```

### Step 3: Let ArgoCD Handle Deployments

ArgoCD will now:
- Monitor the Git repository for changes
- Automatically sync when you push to `develop` or `main` branches
- Deploy updates to the respective environments
- Maintain desired state through continuous reconciliation

For managed resources, make deployment changes through Git and let ArgoCD reconcile the cluster state.

## Environment Configuration

### Development Environment
- **Branch**: `develop`
- **Namespace**: `nebulift-dev`
- **Replicas**: 2 training pods
- **Resources**: 1Gi memory, 1 CPU per pod
- **Configuration**: Debug logging, fast training cycles

### Production Environment
- **Branch**: `main`
- **Namespace**: `nebulift-training`
- **Replicas**: 8 training pods
- **Resources**: 4Gi memory, 3 CPU per pod
- **Configuration**: Optimized for performance and stability

## Directory Structure

```
argocd/
├── applications/
│   ├── nebulift-training.yaml      # Production application
│   └── nebulift-training-dev.yaml  # Development application
├── projects/
│   └── nebulift-project.yaml       # RBAC and permissions
├── applicationset.yaml             # Multi-environment management
└── namespaces.yaml                 # Namespace definitions

k8s/
├── base/                           # Common resources
│   ├── kustomization.yaml
│   ├── configmap.yaml
│   ├── service.yaml
│   └── training-job.yaml
└── overlays/
    ├── development/                # Dev-specific overrides
    │   ├── kustomization.yaml
    │   ├── namespace.yaml
    │   ├── training-dev.conf
    │   └── training-job-patch.yaml
    └── production/                 # Prod-specific overrides
        ├── kustomization.yaml
        ├── namespace.yaml
        ├── training-prod.conf
        ├── training-job-patch.yaml
        └── configmap-patch.yaml
```

## Deployment Operations

### View Application Status

```bash
# List all Nebulift applications
argocd app list | grep nebulift

# Get detailed status
argocd app get nebulift-production

# Watch sync progress
argocd app get nebulift-production --watch
```

### Monitor Applications

```bash
# View application logs
argocd app logs nebulift-production

# View sync history
argocd app history nebulift-production

# Check health status
argocd app get nebulift-production --output json | jq '.status.health'
```

### Refresh Applications

Force ArgoCD to check for changes immediately:

```bash
# Refresh from Git
argocd app get nebulift-production --refresh

# Hard refresh (re-compare with cluster)
argocd app get nebulift-production --hard-refresh
```

### Rollback to Previous Version

ArgoCD tracks all Git commits, making rollbacks simple:

```bash
# View deployment history
argocd app history nebulift-production

# Rollback to previous version
argocd app rollback nebulift-production

# Rollback to specific Git revision
argocd app rollback nebulift-production 5

# Or simply revert the Git commit and push
git revert HEAD
git push origin main
# ArgoCD will automatically sync the revert
```

### Pause Auto-Sync

Temporarily disable auto-sync for maintenance:

```bash
# Disable auto-sync
argocd app set nebulift-production --sync-policy none

# Re-enable auto-sync
argocd app set nebulift-production --sync-policy automated
```

## GitOps Workflow

### How It Works

1. **Developer commits changes** to `develop` or `main` branch
2. **ArgoCD detects changes** in the Git repository
3. **Auto-sync applies changes** to the cluster (if enabled)
4. **ArgoCD maintains state** and self-heals if drift occurs

### Making Configuration Changes

All changes are made through Git - no direct kubectl apply needed:

#### Update Training Parameters

1. Edit the configuration file:
   - **Development**: `k8s/overlays/development/training-dev.conf`
   - **Production**: `k8s/overlays/production/training-prod.conf`

2. Commit and push to the appropriate branch:
   ```bash
   git add k8s/overlays/production/training-prod.conf
   git commit -m "Update learning rate for production"
   git push origin main
   ```

3. **ArgoCD automatically syncs** within ~3 minutes (or instantly with webhooks)

#### Scale Training Pods

1. Update the kustomization file:
   ```yaml
   # k8s/overlays/production/kustomization.yaml
   replicas:
     - name: nebulift-training
       count: 12  # Scale to 12 pods
   ```

2. Commit and push:
   ```bash
   git add k8s/overlays/production/kustomization.yaml
   git commit -m "Scale production training to 12 pods"
   git push origin main
   ```

3. **ArgoCD automatically applies** the change

#### Update Container Image

1. Update the image tag:
   ```yaml
   # k8s/overlays/production/kustomization.yaml
   images:
     - name: ghcr.io/taco-ops/nebulift
       newTag: v1.2.3
   ```

2. Commit and push:
   ```bash
   git add k8s/overlays/production/kustomization.yaml
   git commit -m "Update to version 1.2.3"
   git push origin main
   ```

3. **ArgoCD deploys the new version** automatically

### Manual Sync (Optional)

If you need immediate deployment instead of waiting for auto-sync:

```bash
# Sync specific application
argocd app sync nebulift-production

# Sync with pruning
argocd app sync nebulift-production --prune

# Hard refresh (re-apply everything)
argocd app sync nebulift-production --force
```

## Security and RBAC

### Project Roles

The `nebulift-project.yaml` defines three roles:

1. **Admin**: Full access to all resources
   - Groups: `nebulift-admins`, `platform-team`
   
2. **Developer**: Can sync and view applications
   - Groups: `nebulift-developers`, `ml-engineers`
   
3. **Viewer**: Read-only access
   - Groups: `nebulift-viewers`, `data-scientists`

### Adding Users to Groups

```bash
# Add user to admin group
argocd proj role add-group nebulift admin nebulift-admins

# Add user to developer group
argocd proj role add-group nebulift developer ml-engineers
```

## Monitoring and Troubleshooting

### Common Issues

#### 1. Sync Failures
```bash
# Check sync status
argocd app get nebulift-production

# View detailed errors
argocd app sync nebulift-production --dry-run
```

#### 2. Resource Conflicts
```bash
# Identify conflicting resources
kubectl get jobs -n nebulift-training
kubectl describe job nebulift-training -n nebulift-training

# Force delete stuck resources
kubectl delete job nebulift-training -n nebulift-training --force --grace-period=0
```

#### 3. Image Pull Errors
```bash
# Check image pull secrets
kubectl get secrets -n nebulift-training
kubectl describe pod -l app=nebulift-training -n nebulift-training
```

### Health Checks

```bash
# Check ArgoCD application health
argocd app get nebulift-production --output json | jq '.status.health'

# Check Kubernetes resources
kubectl get all -n nebulift-training
kubectl get events -n nebulift-training --sort-by=.lastTimestamp
```

### Metrics and Monitoring

ArgoCD provides built-in metrics and can integrate with Prometheus:

```bash
# Port forward to ArgoCD metrics
kubectl port-forward svc/argocd-metrics -n argocd 8082:8082

# View metrics
curl http://localhost:8082/metrics
```

## Backup and Disaster Recovery

### Backup ArgoCD Applications

```bash
# Export all applications
argocd app list -o yaml > nebulift-apps-backup.yaml

# Export specific application
argocd app get nebulift-production -o yaml > nebulift-prod-backup.yaml
```

### Restore Applications

```bash
# Restore from backup
kubectl apply -f nebulift-apps-backup.yaml
```

## Advanced Configuration

### Custom Sync Waves

Resources are deployed in order using sync waves:

- Wave 0: Namespaces and RBAC
- Wave 1: Applications and workloads

### Ignore Differences

Common ignored fields for Kubernetes Jobs:

```yaml
ignoreDifferences:
  - group: batch
    kind: Job
    name: nebulift-training
    jsonPointers:
      - /spec/selector
      - /spec/template/metadata/labels
```

### Sync Windows

Prevent automatic syncs during maintenance:

```yaml
syncWindows:
  - kind: deny
    schedule: '0 2 * * 1-5'  # 2-3 AM weekdays
    duration: 1h
    applications: ['*']
    manualSync: false
```

## Integration with CI/CD

CircleCI is the primary CI system for Nebulift. Keep Argo CD deployment changes Git-driven: CI should validate manifests and tests before changes merge, while Argo CD reconciles the cluster from the merged Git state.

Recommended CI checks before relying on automated sync:

- Python lint, type checks, and tests
- `kubectl kustomize k8s/overlays/development`
- `kubectl kustomize k8s/overlays/production`
- Container build checks if deployment images are being updated

## Support and Contributing

For issues with the ArgoCD deployment:

1. Check the [troubleshooting section](#monitoring-and-troubleshooting)
2. Review ArgoCD logs: `kubectl logs -n argocd deployment/argocd-application-controller`
3. Check application events: `argocd app get nebulift-production --show-events`
4. Open an issue in the [Nebulift repository](https://github.com/taco-ops/nebulift/issues)

## References

- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Kustomize Documentation](https://kustomize.io/)
- [Nebulift Architecture Overview](/.github/copilot-instructions.md)
- [Kubernetes Jobs Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
