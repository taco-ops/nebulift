"""
Model Persistence and Checkpoint Management

This module handles saving, loading, and versioning of trained models
with comprehensive metadata and training state.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch

if TYPE_CHECKING:
    from .ml_model import AstroQualityClassifier, ModelTrainer

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Manages model checkpointing with metadata and versioning."""

    @staticmethod
    def save_model(
        trainer: "ModelTrainer",
        path: Union[str, Path],
        include_metadata: bool = True,
        include_training_state: bool = True,
    ) -> None:
        """
        Save model with comprehensive metadata and training history.

        Args:
            trainer: ModelTrainer instance to save
            path: Path to save the model
            include_metadata: Whether to include training metadata
            include_training_state: Whether to save optimizer/scheduler state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare model state dict
        model_data: Dict[str, Any] = {
            "model_state_dict": trainer.model.state_dict(),
            "model_class": trainer.model.__class__.__name__,
            "model_config": {
                "num_classes": trainer.model.backbone.fc.out_features,
                "dropout_rate": trainer.model.dropout.p,
            },
        }

        if include_training_state:
            model_data.update(
                {
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                },
            )

        if include_metadata:
            model_data.update(
                {
                    "training_history": {
                        "train_losses": trainer.train_losses,
                        "val_losses": trainer.val_losses,
                        "train_accuracies": trainer.train_accuracies,
                        "val_accuracies": trainer.val_accuracies,
                    },
                    "metadata": {
                        "save_date": datetime.datetime.now().isoformat(),
                        "pytorch_version": torch.__version__,
                        "device": str(trainer.device),
                        "total_epochs": len(trainer.train_losses),
                        "best_val_accuracy": (
                            max(trainer.val_accuracies)
                            if trainer.val_accuracies
                            else 0.0
                        ),
                        "final_train_loss": (
                            trainer.train_losses[-1] if trainer.train_losses else 0.0
                        ),
                        "final_val_loss": (
                            trainer.val_losses[-1] if trainer.val_losses else 0.0
                        ),
                    },
                },
            )

        torch.save(model_data, path)
        logger.info(f"Model saved successfully to {path}")

        # Save human-readable metadata
        if include_metadata:
            metadata_path = path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "model_info": model_data["metadata"],
                        "training_config": {
                            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                            "weight_decay": trainer.optimizer.param_groups[0][
                                "weight_decay"
                            ],
                            "scheduler_step_size": trainer.scheduler.step_size,
                            "scheduler_gamma": trainer.scheduler.gamma,
                        },
                        "performance_summary": {
                            "best_validation_accuracy": model_data["metadata"][
                                "best_val_accuracy"
                            ],
                            "final_training_loss": model_data["metadata"][
                                "final_train_loss"
                            ],
                            "final_validation_loss": model_data["metadata"][
                                "final_val_loss"
                            ],
                            "epochs_trained": model_data["metadata"]["total_epochs"],
                        },
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Metadata saved to {metadata_path}")

    @staticmethod
    def load_model(
        path: Union[str, Path],
        device: str = "cpu",
        load_training_state: bool = False,
    ) -> "ModelTrainer":
        """
        Load model from saved checkpoint.

        Args:
            path: Path to saved model
            device: Device to load model on
            load_training_state: Whether to restore optimizer/scheduler state

        Returns:
            ModelTrainer instance with loaded model
        """
        from .ml_model import AstroQualityClassifier, ModelTrainer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model data
        checkpoint = torch.load(path, map_location=device)

        # Create model instance
        if checkpoint["model_class"] == "AstroQualityClassifier":
            model = AstroQualityClassifier(
                num_classes=checkpoint["model_config"]["num_classes"],
                dropout_rate=checkpoint["model_config"]["dropout_rate"],
                pretrained=False,  # Don't load pretrained weights when restoring
            )
        else:
            raise ValueError(f"Unknown model class: {checkpoint['model_class']}")

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create trainer
        trainer = ModelTrainer(model, device=device)

        # Restore training state if available and requested
        if load_training_state and "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_training_state and "scheduler_state_dict" in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training history
        if "training_history" in checkpoint:
            history = checkpoint["training_history"]
            trainer.train_losses = history.get("train_losses", [])
            trainer.val_losses = history.get("val_losses", [])
            trainer.train_accuracies = history.get("train_accuracies", [])
            trainer.val_accuracies = history.get("val_accuracies", [])

        logger.info(f"Model loaded successfully from {path}")
        if "metadata" in checkpoint:
            metadata = checkpoint["metadata"]
            logger.info(f"  - Saved: {metadata['save_date']}")
            logger.info(f"  - Epochs: {metadata['total_epochs']}")
            logger.info(f"  - Best accuracy: {metadata['best_val_accuracy']:.2f}%")

        return trainer  # type: ignore[return-value]

    @staticmethod
    def load_model_for_inference(
        path: Union[str, Path],
        device: str = "cpu",
    ) -> "AstroQualityClassifier":
        """
        Load model for inference only (no training state).

        Args:
            path: Path to saved model
            device: Device to load model on

        Returns:
            Model ready for inference
        """
        from .ml_model import AstroQualityClassifier

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=device)

        # Create model instance
        if checkpoint["model_class"] == "AstroQualityClassifier":
            model = AstroQualityClassifier(
                num_classes=checkpoint["model_config"]["num_classes"],
                dropout_rate=checkpoint["model_config"]["dropout_rate"],
                pretrained=False,
            )
        else:
            raise ValueError(f"Unknown model class: {checkpoint['model_class']}")

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()  # Set to inference mode

        logger.info(f"Model loaded for inference from {path}")

        return model  # type: ignore[return-value]

    @staticmethod
    def get_model_info(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get model information without loading the full model.

        Args:
            path: Path to saved model

        Returns:
            Dictionary with model metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        info = {
            "model_class": checkpoint.get("model_class", "Unknown"),
            "model_config": checkpoint.get("model_config", {}),
        }

        if "metadata" in checkpoint:
            info.update(checkpoint["metadata"])

        # Try to load JSON metadata if available
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                json_metadata = json.load(f)
                info["detailed_metadata"] = json_metadata

        return info


class ModelVersioning:
    """Handles model versioning and model registry."""

    def __init__(self, models_dir: Union[str, Path]):
        """
        Initialize model versioning.

        Args:
            models_dir: Directory to store versioned models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "model_registry.json"

    def save_versioned_model(
        self,
        trainer: "ModelTrainer",
        model_name: str,
        version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save model with automatic versioning.

        Args:
            trainer: ModelTrainer instance
            model_name: Name for the model
            version: Version string (auto-generated if not provided)
            tags: Additional tags for the model

        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = self.models_dir / f"{model_name}_v{version}.pth"

        # Save model
        ModelCheckpoint.save_model(trainer, model_path)

        # Update registry
        self._update_registry(model_name, version, str(model_path), tags or {})

        logger.info(f"Model saved with version: {model_name}_v{version}")
        return str(model_path)

    def load_latest_model(self, model_name: str, device: str = "cpu") -> "ModelTrainer":
        """
        Load the latest version of a named model.

        Args:
            model_name: Name of the model to load
            device: Device to load on

        Returns:
            ModelTrainer with latest model
        """
        registry = self._load_registry()

        if model_name not in registry:
            raise ValueError(f"Model {model_name} not found in registry")

        versions = registry[model_name]["versions"]
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")

        # Get latest version (assuming versions are sorted by timestamp)
        latest_version = max(versions.keys())
        model_path = versions[latest_version]["path"]

        logger.info(f"Loading latest model: {model_name}_v{latest_version}")
        return ModelCheckpoint.load_model(model_path, device)

    def list_models(self) -> Dict[str, Any]:
        """
        List all registered models.

        Returns:
            Dictionary with model information
        """
        return self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load model registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                registry_data: dict[str, Any] = json.load(f)
                return registry_data
        return {}

    def _update_registry(
        self,
        model_name: str,
        version: str,
        path: str,
        tags: Dict[str, str],
    ) -> None:
        """Update model registry with new version."""
        registry = self._load_registry()

        if model_name not in registry:
            registry[model_name] = {
                "created": datetime.datetime.now().isoformat(),
                "versions": {},
            }

        registry[model_name]["versions"][version] = {
            "path": path,
            "created": datetime.datetime.now().isoformat(),
            "tags": tags,
        }

        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)
