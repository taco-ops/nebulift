"""
Distributed Training Module for Kubernetes Clusters

Provides distributed training capabilities for Nebulift on Raspberry Pi 5
Kubernetes clusters using PyTorch's CPU-only distributed training.
"""

__version__ = "0.1.0"
__author__ = "David Perez"

from .data_sharding import create_distributed_dataloaders
from .k8s_trainer import K8sDistributedTrainer
from .model_aggregation import aggregate_model_updates

__all__ = [
    "K8sDistributedTrainer",
    "create_distributed_dataloaders",
    "aggregate_model_updates",
]
