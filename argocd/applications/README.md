# Individual Application Files

**⚠️ IMPORTANT**: These individual Application files are **NOT needed** if you're using the ApplicationSet (`../applicationset.yaml`).

## Current Setup

The ApplicationSet automatically creates and manages:
- `nebulift-development` (from develop branch → nebulift-dev namespace)
- `nebulift-production` (from main branch → nebulift-training namespace)

## When to Use These Files

Use these individual Application files ONLY if:
- You want to manage applications individually (not recommended)
- You're testing a specific configuration
- You want to disable the ApplicationSet and use manual application management

## How to Use

If you choose to use individual applications instead of the ApplicationSet:

1. **Delete the ApplicationSet**:
   ```bash
   kubectl delete applicationset nebulift-environments -n argocd
   ```

2. **Apply individual applications**:
   ```bash
   kubectl apply -f argocd/applications/nebulift-training-dev.yaml
   kubectl apply -f argocd/applications/nebulift-training.yaml
   ```

## Recommendation

**Use the ApplicationSet** - it's the GitOps best practice for managing multiple environments from a single source.
