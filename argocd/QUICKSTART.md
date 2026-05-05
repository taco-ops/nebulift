# Nebulift Argo CD Quick Start

## Purpose

This guide describes the shortest path to register the Nebulift Kubernetes resources with Argo CD. Use it after Argo CD is already installed and reachable from your workstation.

## One-Time Setup

1. Log in to Argo CD:

   ```bash
   argocd login argocd.spacedojo.local --insecure
   ```

2. Apply the Nebulift Argo CD resources from the repository root:

   ```bash
   kubectl apply -f argocd/projects/nebulift-project.yaml
   kubectl apply -f argocd/applicationset.yaml
   ```

3. Verify that applications were created:

   ```bash
   argocd app list | grep nebulift
   ```

The ApplicationSet creates both `nebulift-development` and `nebulift-production`. Do not also apply the individual files in `argocd/applications/` unless you are intentionally replacing the ApplicationSet model.

## Environment Mapping

- `develop` branch deploys to the `nebulift-dev` namespace.
- `main` branch deploys to the `nebulift-training` namespace.

## Daily Workflow

Use Git as the source of truth for deployment changes:

```bash
vim k8s/overlays/production/training-prod.conf
git add k8s/overlays/production/training-prod.conf
git commit -m "Update training parameters"
git push origin main
```

Argo CD detects the Git change and reconciles the target environment according to the ApplicationSet configuration.

## View Applications

```bash
argocd app list | grep nebulift
argocd app get nebulift-production
```

If using the web UI, open your Argo CD endpoint and inspect the `nebulift-development` or `nebulift-production` application.

## Common Operations

```bash
# Watch production sync status
argocd app get nebulift-production --watch

# Force an immediate sync
argocd app sync nebulift-production

# Refresh from Git
argocd app get nebulift-production --refresh

# View logs
argocd app logs nebulift-production

# View deployment history
argocd app history nebulift-production

# Roll back using Argo CD history
argocd app rollback nebulift-production
```

## Temporarily Disable Auto-Sync

```bash
argocd app set nebulift-production --sync-policy none
argocd app sync nebulift-production
argocd app set nebulift-production --sync-policy automated
```

Prefer changing Git and letting Argo CD reconcile normally. Use manual sync operations only for controlled maintenance or troubleshooting.

## Check Pending Changes

```bash
argocd app diff nebulift-production
```

## Next Steps

- Read the full guide: [argocd/README.md](./README.md)
- Configure webhooks if faster reconciliation is required
- Configure notifications for failed syncs or degraded health
- Add render and policy checks to CI before enabling production automation
