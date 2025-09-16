"""
Model Aggregation for Distributed Training

Handles model synchronization and aggregation across Kubernetes
nodes during distributed training.
"""

import logging
from typing import TYPE_CHECKING, Dict

import torch
import torch.distributed as dist
from torch import nn

if TYPE_CHECKING:
    from .k8s_trainer import K8sDistributedTrainer

logger = logging.getLogger(__name__)


def aggregate_model_updates(model: nn.Module, world_size: int) -> None:
    """
    Aggregate model parameters across all nodes using all-reduce.

    Args:
        model: PyTorch model to aggregate
        world_size: Number of distributed nodes
    """
    if world_size <= 1:
        return

    logger.debug("Starting model parameter aggregation")

    # Aggregate all model parameters
    for param in model.parameters():
        if param.grad is not None:
            # Average gradients across all nodes
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def collect_metrics_from_all_nodes(
    local_metrics: Dict[str, float],
    world_size: int,
    rank: int,
) -> Dict[str, float]:
    """
    Collect and average metrics from all nodes.

    Args:
        local_metrics: Metrics from current node
        world_size: Total number of nodes
        rank: Current node rank

    Returns:
        Dictionary of averaged metrics
    """
    if world_size <= 1:
        return local_metrics

    aggregated_metrics = {}

    for metric_name, value in local_metrics.items():
        # Convert to tensor for distributed operations
        value_tensor = torch.tensor(value, dtype=torch.float32)

        # Sum across all nodes
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)

        # Calculate average
        averaged_value = (value_tensor / world_size).item()
        aggregated_metrics[metric_name] = averaged_value

    if rank == 0:
        logger.info(f"Aggregated metrics: {aggregated_metrics}")

    return aggregated_metrics


def broadcast_model_state(model: nn.Module, src_rank: int = 0) -> None:
    """
    Broadcast model state from source rank to all other nodes.

    Args:
        model: PyTorch model to broadcast
        src_rank: Source rank to broadcast from (usually 0)
    """
    logger.debug(f"Broadcasting model state from rank {src_rank}")

    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)


def synchronize_training_state(
    trainer: "K8sDistributedTrainer",
    epoch: int,
    best_loss: float,
    world_size: int,
    rank: int,
) -> Dict[str, float]:
    """
    Synchronize training state across all nodes.

    Args:
        trainer: Distributed trainer instance
        epoch: Current epoch
        best_loss: Best validation loss
        world_size: Number of nodes
        rank: Current node rank

    Returns:
        Dictionary with synchronized state
    """
    if world_size <= 1:
        return {"epoch": epoch, "best_loss": best_loss}

    # Create tensors for synchronization
    state_tensor = torch.tensor([epoch, best_loss], dtype=torch.float32)

    # Broadcast state from rank 0
    dist.broadcast(state_tensor, src=0)

    synced_epoch, synced_best_loss = state_tensor.tolist()

    return {"epoch": int(synced_epoch), "best_loss": synced_best_loss}


def wait_for_all_nodes(timeout: float = 300.0) -> None:
    """
    Wait for all nodes to reach this point before continuing.

    Args:
        timeout: Maximum time to wait in seconds
    """
    if dist.is_initialized():
        logger.debug("Waiting for all nodes to synchronize")
        dist.barrier()
        logger.debug("All nodes synchronized")


def check_distributed_health(world_size: int, rank: int) -> bool:
    """
    Check if distributed training is healthy across all nodes.

    Args:
        world_size: Expected number of nodes
        rank: Current node rank

    Returns:
        True if all nodes are healthy
    """
    if world_size <= 1:
        return True

    try:
        # Simple health check - all nodes contribute 1, sum should equal world_size
        health_tensor = torch.tensor(1.0)
        dist.all_reduce(health_tensor, op=dist.ReduceOp.SUM)

        expected_sum = float(world_size)
        actual_sum = health_tensor.item()

        is_healthy = abs(actual_sum - expected_sum) < 0.1

        if rank == 0:
            logger.info(
                f"Distributed health check: {actual_sum}/{expected_sum} "
                f"nodes responding ({'HEALTHY' if is_healthy else 'UNHEALTHY'})",
            )

        return bool(is_healthy)  # type: ignore[return-value]

    except Exception as e:
        logger.error(f"Distributed health check failed: {e}")
        return False
