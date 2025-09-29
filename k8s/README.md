# Distributed Training Deployment Guide

## Overview
This guide shows how to deploy distributed training of your IC 14 FITS dataset (1,469 files, 5.7GB) across your K3s Raspberry Pi cluster using NFS shared storage.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Master Node   │    │  Worker Node 1  │    │  Worker Node 2  │
│                 │    │                 │    │                 │
│ NFS Server      │◄───┤ Training Pod 1  │    │ Training Pod 2  │
│ /mnt/cluster-   │    │ (NFS Client)    │    │ (NFS Client)    │
│ storage/        │    │                 │    │                 │
│ ├─ nebulift/    │    └─────────────────┘    └─────────────────┘
│ │  ├─ data/     │              │                        │
│ │  │  └─ raw/   │              │      NFS Network       │
│ │  │     └─fits/│◄─────────────┼────────────────────────┘
│ │  └─ models/   │              │
│ └─ Training Pod │              │
│    Coordinator  │              │
└─────────────────┘              │
```

## Quick Start

### 1. Transfer FITS Data
```bash
# Run on master node (where NFS is mounted)
../transfer-data-to-nfs.sh
```

### 2. Deploy Everything
```bash
# Deploy NFS server + distributed training
../deploy-distributed-training.sh
```

### 3. Monitor Training
```bash
# Watch training logs
kubectl logs -f job/nebulift-training-nfs

# Check pod distribution
kubectl get pods -l app=nebulift-training -o wide
```

## Files Overview

### Data Transfer Scripts
- `../transfer-data-to-nfs.sh` - Simple script to copy FITS data to NFS storage
- `../setup-cluster-training.sh` - Comprehensive setup with K8s integration
- `../deploy-distributed-training.sh` - Complete deployment automation

### Kubernetes Configurations
- `nfs-server.yaml` - NFS server deployment on master node
- `training-job.yaml` - Original training job (updated for NFS)
- `training-job-nfs.yaml` - Dedicated NFS-based training job
- `nfs-daemonset.yaml` - Alternative: Mount NFS on all nodes
- `configmap.yaml` - Training configuration
- `service.yaml` - Service discovery
- `pvc.yaml` - Persistent volume claims (legacy)

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
