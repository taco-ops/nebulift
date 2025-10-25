# Nebulift ArgoCD Quick Start

## TL;DR - GitOps Deployment

### One-Time Setup (5 minutes)

1. **Login to ArgoCD**:
   ```bash
   argocd login argocd.spacedojo.local --insecure
   ```

2. **Deploy ArgoCD resources** (one time only):
   ```bash
   # From the repo root
   kubectl apply -f argocd/projects/nebulift-project.yaml
   kubectl apply -f argocd/applicationset.yaml
   ```

3. **Done!** ArgoCD now manages your deployments.

### Daily Workflow

**To deploy changes, just use Git:**

```bash
# Make your changes
vim k8s/overlays/production/training-prod.conf

# Commit and push
git add .
git commit -m "Update training parameters"
git push origin main

# ArgoCD automatically deploys within ~3 minutes
```

**No kubectl apply needed!** ArgoCD watches Git and deploys automatically.

## What Was Created?

After applying the ArgoCD resources, you have:

✅ **Development environment** (`nebulift-dev` namespace)
- Watches: `develop` branch
- 2 training pods
- Auto-syncs on every commit to `develop`

✅ **Production environment** (`nebulift-training` namespace)  
- Watches: `main` branch
- 8 training pods
- Auto-syncs on every commit to `main`

## View Your Applications

```bash
# CLI
argocd app list | grep nebulift
argocd app get nebulift-production

# Web UI
# Open: https://argocd.spacedojo.local
```

## Making Changes

### Update Training Configuration

```bash
# Edit config
vim k8s/overlays/production/training-prod.conf

# Commit and push
git commit -am "Increase batch size to 128"
git push origin main

# Watch ArgoCD deploy
argocd app get nebulift-production --watch
```

### Scale Training Pods

```bash
# Edit kustomization
vim k8s/overlays/production/kustomization.yaml
# Change: count: 12

# Commit and push
git commit -am "Scale to 12 pods"
git push origin main
```

### Update Container Image

```bash
# Edit kustomization
vim k8s/overlays/production/kustomization.yaml
# Change: newTag: v1.2.3

# Commit and push  
git commit -am "Deploy version 1.2.3"
git push origin main
```

## Common Commands

```bash
# View status
argocd app list
argocd app get nebulift-production

# Force immediate sync (don't wait for auto-sync)
argocd app sync nebulift-production

# Refresh from Git
argocd app get nebulift-production --refresh

# View logs
argocd app logs nebulift-production

# Rollback
argocd app history nebulift-production
argocd app rollback nebulift-production
```

## Troubleshooting

### Sync is stuck or failed
```bash
argocd app get nebulift-production
argocd app sync nebulift-production --force --prune
```

### Want to disable auto-sync temporarily
```bash
argocd app set nebulift-production --sync-policy none
# Make changes, test manually
argocd app sync nebulift-production
# Re-enable
argocd app set nebulift-production --sync-policy automated
```

### Check what ArgoCD will deploy
```bash
argocd app diff nebulift-production
```

## Key Points

🎯 **GitOps = No manual kubectl**
- Commit to Git → ArgoCD deploys
- Your cluster state always matches Git

🔄 **Auto-sync enabled**
- Changes deploy within ~3 minutes
- Or instantly with Git webhooks

🌲 **Branch = Environment**
- `develop` → development env
- `main` → production env

📊 **Observability**
- ArgoCD UI shows deployment status
- Complete audit trail in Git history

## Next Steps

- Read full docs: [argocd/README.md](./README.md)
- Configure webhooks for instant sync
- Set up notifications for deployment events
- Integrate with CI/CD pipeline