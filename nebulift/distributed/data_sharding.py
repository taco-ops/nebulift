"""
Data Sharding for Distributed Training

Handles data distribution across Kubernetes nodes for efficient
parallel training on Raspberry Pi 5 clusters.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..ml_model import AstroImageDataset, create_data_transforms

logger = logging.getLogger(__name__)


def create_distributed_dataloaders(
    image_paths: list[str],
    labels: List[int],
    batch_size: int = 32,
    val_split: float = 0.2,
    rank: int = 0,
    world_size: int = 1,
    fits_processor=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create distributed data loaders for training across K8s nodes.

    Args:
        image_paths: List of paths to training images
        labels: List of corresponding labels
        batch_size: Batch size per node
        val_split: Fraction of data for validation
        rank: Current node rank
        world_size: Total number of nodes
        fits_processor: FITS processor instance

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info(f"Creating distributed dataloaders for rank {rank}/{world_size}")

    # Split data into train/validation
    num_samples = len(image_paths)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    # Create indices for splitting
    indices = torch.randperm(num_samples).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Create training dataset
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    train_dataset = AstroImageDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=create_data_transforms(train=True),
        fits_processor=fits_processor,
    )

    # Create validation dataset
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    val_dataset = AstroImageDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=create_data_transforms(train=False),
        fits_processor=fits_processor,
    )

    # Create distributed samplers
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        if world_size > 1
        else None
    )

    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if world_size > 1
        else None
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=2,  # Limit workers for RPi5
        pin_memory=False,  # CPU-only training
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=1,  # Fewer workers for validation
        pin_memory=False,
        drop_last=False,
    )

    logger.info(
        f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}",
    )

    return train_loader, val_loader


def shard_fits_files(
    data_dir: Path,
    world_size: int,
    rank: int,
    file_pattern: str = "*.fits",
) -> List[Path]:
    """
    Shard FITS files across distributed nodes.

    Args:
        data_dir: Directory containing FITS files
        world_size: Total number of nodes
        rank: Current node rank
        file_pattern: File pattern to match

    Returns:
        List of FITS files assigned to this node
    """
    all_files = sorted(list(data_dir.glob(file_pattern)))

    # Simple round-robin sharding
    node_files = [f for i, f in enumerate(all_files) if i % world_size == rank]

    logger.info(f"Node {rank} assigned {len(node_files)}/{len(all_files)} files")

    return node_files


def balance_dataset_across_nodes(
    image_paths: list[str],
    labels: List[int],
    world_size: int,
) -> Tuple[list[str], List[int]]:
    """
    Balance dataset to ensure equal class distribution across nodes.

    Args:
        image_paths: All image paths
        labels: All labels
        world_size: Number of nodes

    Returns:
        Tuple of (balanced_paths, balanced_labels)
    """
    from collections import defaultdict

    # Group by class
    class_groups = defaultdict(list)
    for path, label in zip(image_paths, labels):
        class_groups[label].append(path)

    # Ensure each class has samples divisible by world_size
    balanced_paths = []
    balanced_labels = []

    for label, paths in class_groups.items():
        # Truncate to make divisible by world_size
        samples_per_node = len(paths) // world_size
        total_samples = samples_per_node * world_size

        selected_paths = paths[:total_samples]
        selected_labels = [label] * total_samples

        balanced_paths.extend(selected_paths)
        balanced_labels.extend(selected_labels)

    logger.info(
        f"Balanced dataset: {len(balanced_paths)} samples across {world_size} nodes",
    )

    return balanced_paths, balanced_labels
