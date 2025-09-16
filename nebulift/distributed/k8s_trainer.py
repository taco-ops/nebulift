"""
Kubernetes Distributed Trainer

Extends the base ModelTrainer to support distributed training across
Raspberry Pi 5 nodes in a Kubernetes cluster using CPU-only PyTorch.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ..ml_model import AstroQualityClassifier

from ..ml_model import ModelTrainer

logger = logging.getLogger(__name__)


class K8sDistributedTrainer(ModelTrainer):
    """Distributed trainer for Kubernetes clusters with CPU-only nodes."""

    def __init__(
        self,
        model: "AstroQualityClassifier",
        learning_rate: float = 0.001,
        backend: str = "gloo",  # CPU-only backend
        master_addr: Optional[str] = None,
        master_port: Optional[str] = None,
    ):
        """
        Initialize distributed trainer.

        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimizer
            backend: Distributed backend ('gloo' for CPU-only)
            master_addr: Master node address (from K8s env if None)
            master_port: Master node port (from K8s env if None)
        """
        # Get distributed training info from Kubernetes environment
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # Set master node info for distributed coordination
        if master_addr is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
        if master_port is None:
            master_port = os.environ.get("MASTER_PORT", "29500")

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        logger.info(
            f"Initializing distributed training: rank={self.rank}, "
            f"world_size={self.world_size}, backend={backend}",
        )

        # Initialize distributed process group
        if self.world_size > 1:
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
            )
            logger.info("Distributed process group initialized")

        # Initialize base trainer
        super().__init__(model, "cpu", learning_rate)

        # Wrap model for distributed training
        if self.world_size > 1:
            self.model = DistributedDataParallel(  # type: ignore[assignment]
                self.model,
                device_ids=None,
                output_device=None,  # CPU-only
            )
            logger.info("Model wrapped with DistributedDataParallel")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch with distributed coordination.

        Args:
            train_loader: DataLoader with DistributedSampler

        Returns:
            Tuple of (average_loss, accuracy)
        """
        # Set epoch for DistributedSampler to ensure proper shuffling
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(len(self.train_losses))

        return super().train_epoch(train_loader)

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        if self.world_size > 1 and dist.is_initialized():
            logger.info("Cleaning up distributed process group")
            dist.destroy_process_group()

    def save_checkpoint(self, filepath: str, epoch: int, best_loss: float) -> None:
        """
        Save model checkpoint (only on rank 0 to avoid conflicts).

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            best_loss: Best validation loss so far
        """
        if self.rank == 0:  # Only main process saves
            # Extract underlying model from DDP wrapper
            model_state = (
                self.model.module.state_dict()
                if hasattr(self.model, "module")
                else self.model.state_dict()
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_loss": best_loss,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_accuracies": self.train_accuracies,
                "val_accuracies": self.val_accuracies,
            }

            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> dict:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location="cpu")

        # Load into underlying model (handle DDP wrapper)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training history
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_accuracies = checkpoint.get("train_accuracies", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])

        logger.info(f"Checkpoint loaded from {filepath}")

        return {"epoch": checkpoint["epoch"], "best_loss": checkpoint["best_loss"]}

    def all_reduce_metrics(self, loss: float, accuracy: float) -> Tuple[float, float]:
        """
        Average metrics across all nodes for consistent reporting.

        Args:
            loss: Local node loss
            accuracy: Local node accuracy

        Returns:
            Tuple of (averaged_loss, averaged_accuracy)
        """
        if self.world_size <= 1:
            return loss, accuracy

        # Convert to tensors for distributed operations
        loss_tensor = torch.tensor(loss, dtype=torch.float32)
        acc_tensor = torch.tensor(accuracy, dtype=torch.float32)

        # Average across all nodes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)

        averaged_loss = (loss_tensor / self.world_size).item()
        averaged_accuracy = (acc_tensor / self.world_size).item()

        return averaged_loss, averaged_accuracy

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
