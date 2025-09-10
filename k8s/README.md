# Kubernetes Deployment Files for Nebulift

This directory contains Kubernetes manifests for deploying Nebulift training jobs on Raspberry Pi 5 clusters.

## Files

- `training-job.yaml` - Distributed training job
- `configmap.yaml` - Configuration for training parameters  
- `pvc.yaml` - Persistent volume for training data
- `service.yaml` - Service for distributed coordination

## Usage

```bash
# Apply all manifests
kubectl apply -f k8s/

# Monitor training job
kubectl logs -f job/nebulift-training

# Check pod status
kubectl get pods -l app=nebulift-training

# Scale training job
kubectl patch job nebulift-training -p '{"spec":{"parallelism":8}}'
```

## Configuration

Edit `configmap.yaml` to adjust training parameters:
- Learning rate
- Batch size  
- Epochs
- Model checkpoint paths
