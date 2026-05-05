# Individual Application Files

These individual Application files are not needed if you use the ApplicationSet (`../applicationset.yaml`).

## Current Setup

The ApplicationSet automatically creates and manages:
- `nebulift-development` from the `develop` branch to the `nebulift-dev` namespace
- `nebulift-production` from the `main` branch to the `nebulift-training` namespace

## When to Use These Files

Use these individual Application files only if:
- You want to manage applications individually (not recommended)
- You're testing a specific configuration
- You want to disable the ApplicationSet and use manual application management

## How to Use

If you choose to use individual applications instead of the ApplicationSet:

1. Delete the ApplicationSet:
   ```bash
   kubectl delete applicationset nebulift-environments -n argocd
   ```

2. Apply individual applications:
   ```bash
   kubectl apply -f argocd/applications/nebulift-training-dev.yaml
   kubectl apply -f argocd/applications/nebulift-training.yaml
   ```

## Recommendation

Use the ApplicationSet for the default multi-environment setup. It keeps environment management centralized and avoids duplicate Application ownership.
