"""
Kubernetes Distributed Trainer

Extends the base ModelTrainer to support distributed training across
Raspberry Pi 5 nodes in a Kubernetes cluster using CPU-only PyTorch.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..ml_model import (
    LABEL_IDS,
    AstroImageDataset,
    AstroQualityClassifier,
    ModelTrainer,
    create_data_transforms,
)
from ..training import collect_class_directory_records

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
            # If rank is 0, use own POD_IP to avoid chicken-and-egg problem
            if self.rank == 0:
                pod_ip = os.environ.get("POD_IP")
                if pod_ip:
                    master_addr = pod_ip
                    logger.info(f"Rank 0 using POD_IP as MASTER_ADDR: {master_addr}")
        if master_port is None:
            master_port = os.environ.get("MASTER_PORT", "29500")

        # For worker nodes, resolve the headless service to get rank 0's IP
        if self.rank != 0 and master_addr == os.environ.get("MASTER_ADDR"):
            # Try to get the IP from the coordinator service
            import socket
            import time

            # Wait for rank 0 to be ready
            max_retries = 30
            for attempt in range(max_retries):
                try:
                    # Try to resolve the headless service + rank 0 pod
                    # Format: <job-name>-<index>.<headless-service>.<namespace>.svc.cluster.local
                    job_name = os.environ.get("POD_NAME", "nebulift-training").rsplit(
                        "-", 2
                    )[0]
                    rank_0_dns = f"{job_name}-0.{master_addr}"
                    logger.info(f"Attempting to resolve rank 0 DNS: {rank_0_dns}")
                    rank_0_ip = socket.gethostbyname(rank_0_dns)
                    master_addr = rank_0_ip
                    logger.info(f"Resolved rank 0 IP: {master_addr}")
                    break
                except socket.gaierror:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Could not resolve rank 0, retrying ({attempt + 1}/{max_retries})..."
                        )
                        time.sleep(2)
                    else:
                        logger.warning(
                            f"Could not resolve rank 0 DNS, using service name: {master_addr}"
                        )

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

    def train_epoch(
        self,
        train_loader: DataLoader,
        batch_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[float, float]:
        """
        Train for one epoch with distributed coordination.

        Args:
            train_loader: DataLoader with DistributedSampler
            batch_callback: Optional callback invoked after each batch with the
                current batch index (1-based) and total batch count.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        # Set epoch for DistributedSampler to ensure proper shuffling
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(len(self.train_losses))

        return super().train_epoch(train_loader, batch_callback=batch_callback)

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


def _build_distributed_loaders(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for the distributed job.

    The training loader is sharded across ranks via ``DistributedSampler``
    so each rank sees a disjoint subset of the training data. The
    validation loader is intentionally **not** sharded: each rank
    evaluates against the full held-out set so per-rank ``val_accuracy``
    figures are directly comparable. This trades extra compute for
    correctness while distributed metric reduction is not yet wired in.
    """
    # Local import to avoid circular dependency on the FITSProcessor at
    # module import time (the package is only present when training).
    from ..fits_processor import FITSProcessor

    fits_processor = FITSProcessor()
    train_dataset = AstroImageDataset(
        [record["path"] for record in train_records],
        [record["label"] for record in train_records],
        transform=create_data_transforms(train=True),
        fits_processor=fits_processor,
    )
    val_dataset = AstroImageDataset(
        [record["path"] for record in val_records],
        [record["label"] for record in val_records],
        transform=create_data_transforms(train=False),
        fits_processor=fits_processor,
    )

    if world_size > 1:
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def _save_unwrapped_checkpoint(
    trainer: "K8sDistributedTrainer", model_output_path: Path
) -> None:
    """Persist the trainer's underlying model via ``ModelCheckpoint``.

    Unwraps the ``DistributedDataParallel`` shell before saving so the
    resulting checkpoint is interchangeable with locally trained
    artifacts (no ``module.`` prefix on tensor keys).
    """
    from ..model_persistence import ModelCheckpoint

    wrapped = trainer.model
    underlying = (
        wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
    )
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        trainer.model = underlying  # type: ignore[assignment]
        ModelCheckpoint.save_model(trainer, model_output_path)
    finally:
        trainer.model = wrapped


def main() -> None:
    """Run a distributed training job inside a Kubernetes pod.

    Reads runtime configuration from the environment (typically populated
    by the training-job ConfigMap):

    - ``TRAIN_DATA_PATH`` / ``VAL_DATA_PATH``: directories containing
      ``clean/``, ``contaminated/``, and ``review/`` subdirectories with
      FITS files.
    - ``MODEL_OUTPUT_PATH``: directory where the final checkpoint is
      written by rank 0 only.
    - ``EPOCHS``, ``BATCH_SIZE``, ``LEARNING_RATE``: training
      hyperparameters.
    - ``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``:
      distributed-training coordination (provided by the K8s Job's
      Indexed-completion plumbing).

    Exits non-zero on any failure so the Job's ``backoffLimit`` can take
    effect.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    train_dir = Path(os.environ.get("TRAIN_DATA_PATH", "/data/train"))
    val_dir = Path(os.environ.get("VAL_DATA_PATH", "/data/val"))
    model_output_dir = Path(os.environ.get("MODEL_OUTPUT_PATH", "/models"))
    model_output_path = model_output_dir / "nebulift_distributed.pth"
    epochs = int(os.environ.get("EPOCHS", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.001"))
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    logger.info(
        "Distributed training start: rank=%d, world_size=%d, train_dir=%s, "
        "val_dir=%s, epochs=%d, batch_size=%d, learning_rate=%f",
        rank,
        world_size,
        train_dir,
        val_dir,
        epochs,
        batch_size,
        learning_rate,
    )

    try:
        train_records = collect_class_directory_records(train_dir)
        val_records = collect_class_directory_records(val_dir)
        if not train_records:
            raise RuntimeError(
                f"No FITS files found under {train_dir}. Expected per-class "
                "subdirectories (clean/, contaminated/, review/)."
            )
        if not val_records:
            raise RuntimeError(
                f"No FITS files found under {val_dir}. Expected per-class "
                "subdirectories (clean/, contaminated/, review/)."
            )

        logger.info(
            "Discovered %d training records and %d validation records",
            len(train_records),
            len(val_records),
        )

        train_loader, val_loader = _build_distributed_loaders(
            train_records,
            val_records,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            num_workers=num_workers,
        )

        model = AstroQualityClassifier(
            num_classes=len(LABEL_IDS),
            pretrained=False,
        )
        trainer = K8sDistributedTrainer(
            model,
            learning_rate=learning_rate,
            backend=os.environ.get("BACKEND", "gloo"),
        )

        try:
            trainer.train(train_loader, val_loader, epochs=epochs)
            if trainer.is_main_process():
                _save_unwrapped_checkpoint(trainer, model_output_path)
                logger.info("Checkpoint persisted to %s", model_output_path)
        finally:
            trainer.cleanup()

        logger.info("Rank %d: training completed successfully", rank)

    except Exception:
        logger.exception("Training failed on rank %d", rank)
        sys.exit(1)


if __name__ == "__main__":
    main()
