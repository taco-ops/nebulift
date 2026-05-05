"""
ResNet18 Model for Astrophotography Quality Assessment

Implements a ResNet18-based neural network for binary classification of
astronomical images as "contaminated", "clean", or "review".
Optimized for CPU-only inference on resource-constrained devices like Raspberry Pi 5.
"""

import csv
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

if TYPE_CHECKING:
    from .fits_processor import FITSProcessor

logger = logging.getLogger(__name__)

LABEL_CONTAMINATED = 0
LABEL_CLEAN = 1
LABEL_REVIEW = 2
LABEL_NAMES = {
    LABEL_CONTAMINATED: "contaminated",
    LABEL_CLEAN: "clean",
    LABEL_REVIEW: "review",
}
LABEL_IDS = {label_name: label_id for label_id, label_name in LABEL_NAMES.items()}


class AstroImageDataset(Dataset):
    """Dataset class for astronomical images with quality labels."""

    def __init__(
        self,
        image_paths: list[Union[str, Path]],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
        fits_processor: Optional["FITSProcessor"] = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            image_paths: List of paths to image files
            labels: List of labels (0=contaminated, 1=clean, 2=review)
            transform: Optional image transforms
            fits_processor: FITSProcessor instance for loading FITS files
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.fits_processor = fits_processor

        assert len(image_paths) == len(labels), "Number of images and labels must match"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single item from the dataset."""
        image_path = self.image_paths[idx]
        image_path_str = str(image_path)
        label = self.labels[idx]

        try:
            # Load image
            if self.fits_processor and image_path_str.endswith(
                (".fits", ".fit", ".fts"),
            ):
                # Load FITS file
                processed_data = self.fits_processor.process_fits_file(
                    Path(image_path_str),
                )
                if processed_data is None:
                    raise ValueError(f"Could not load FITS file: {image_path}")

                # Get ML-ready image
                image_array = processed_data["ml_image"]

                # Convert to PIL Image for transforms
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))

                # Convert grayscale to RGB for ResNet
                if image_pil.mode != "RGB":
                    image_pil = image_pil.convert("RGB")

            else:
                # Load regular image file
                image_pil = Image.open(image_path)
                if image_pil.mode != "RGB":
                    image_pil = image_pil.convert("RGB")

            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image_pil)
            else:
                # Default transform
                transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                    ],
                )
                image_tensor = transform(image_pil)

            return image_tensor, label

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a dummy tensor and label
            dummy_tensor = torch.zeros((3, 224, 224))
            return dummy_tensor, 0


class AstroQualityClassifier(nn.Module):
    """ResNet18-based classifier for astrophotography quality assessment."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the classifier.

        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use pretrained ResNet18 weights
            dropout_rate: Dropout rate for regularization
        """
        super(AstroQualityClassifier, self).__init__()

        # Load ResNet18 backbone
        self.backbone = resnet18(pretrained=pretrained)

        # Modify final layer for our classification task
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Extract features using ResNet backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.fc(x)

        return x


class ModelTrainer:
    """Handles training and evaluation of the astrophotography quality model."""

    def __init__(
        self,
        model: AstroQualityClassifier,
        device: str = "cpu",
        learning_rate: float = 1e-4,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1,
        )

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 10 == 0:
                logger.info(f"Train Batch: {batch_idx}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def evaluate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        save_path: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save the best model

        Returns:
            Dictionary containing training history
        """
        best_val_accuracy = 0.0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_accuracy and save_path:
                best_val_accuracy = val_acc
                torch.save(self.model.state_dict(), save_path)
                logger.info(
                    f"Saved best model with validation accuracy: {val_acc:.2f}%",
                )

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

    def save_model(
        self,
        path: Union[str, Path],
        include_metadata: bool = True,
        include_training_state: bool = True,
    ) -> None:
        """
        Save model with comprehensive metadata and training history.

        Args:
            path: Path to save the model
            include_metadata: Whether to include training metadata
            include_training_state: Whether to save optimizer/scheduler state
        """
        from .model_persistence import ModelCheckpoint

        ModelCheckpoint.save_model(self, path, include_metadata, include_training_state)

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
        from .model_persistence import ModelCheckpoint

        return ModelCheckpoint.load_model(path, device, load_training_state)  # type: ignore[return-value]

    def get_training_history(self) -> dict[str, list[float]]:
        """
        Get current training history.

        Returns:
            Dictionary with training metrics
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }


class QualityPredictor:
    """Handles inference for quality prediction on new images."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model weights
            device: Device for inference
        """
        from .model_persistence import ModelCheckpoint

        self.device = device

        # Use ModelCheckpoint for proper loading
        try:
            self.model = ModelCheckpoint.load_model_for_inference(model_path, device)
        except Exception as e:
            # Fallback for legacy formats
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    # New format with metadata
                    config = checkpoint.get("model_config", {})
                    self.model = AstroQualityClassifier(
                        num_classes=config.get("num_classes", 2),
                        dropout_rate=config.get("dropout_rate", 0.2),
                        pretrained=False,
                    )
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    # Old format - direct state dict
                    self.model = AstroQualityClassifier(num_classes=2, pretrained=False)
                    self.model.load_state_dict(checkpoint)

                self.model.to(device)
                self.model.eval()
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to load model from {model_path}: {e}. Fallback also failed: {fallback_error}"
                )

        # Ensure model is on correct device and in eval mode
        self.model.to(device)
        self.model.eval()

        # Define transforms for inference
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

    def predict_single(
        self,
        image_path: str,
        fits_processor: Optional["FITSProcessor"] = None,
    ) -> dict[str, Any]:
        """
        Predict quality for a single image.

        Args:
            image_path: Path to image file
            fits_processor: Optional FITSProcessor for FITS files

        Returns:
            Dictionary with prediction results
        """
        try:
            # Load and preprocess image
            if fits_processor and image_path.endswith((".fits", ".fit", ".fts")):
                processed_data = fits_processor.process_fits_file(Path(image_path))
                if processed_data is None:
                    raise ValueError(f"Could not load FITS file: {image_path}")

                image_array = processed_data["ml_image"]
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
                if image_pil.mode != "RGB":
                    image_pil = image_pil.convert("RGB")
            else:
                image_pil = Image.open(image_path)
                if image_pil.mode != "RGB":
                    image_pil = image_pil.convert("RGB")

            # Apply transforms
            image_tensor = self.transform(image_pil).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)

                predicted_class = int(torch.argmax(output, dim=1).item())
                class_probabilities = {
                    LABEL_NAMES.get(index, f"class_{index}"): probabilities[0][
                        index
                    ].item()
                    for index in range(probabilities.shape[1])
                }
                contaminated_prob = class_probabilities.get("contaminated", 0.0)
                clean_prob = class_probabilities.get("clean", 0.0)
                review_prob = class_probabilities.get("review", 0.0)
                confidence = max(class_probabilities.values())

            return {
                "predicted_class": predicted_class,
                "predicted_label": LABEL_NAMES.get(
                    predicted_class, str(predicted_class)
                ),
                "confidence": confidence,
                "contaminated_probability": contaminated_prob,
                "clean_probability": clean_prob,
                "review_probability": review_prob,
                "class_probabilities": class_probabilities,
                "is_clean": predicted_class == LABEL_CLEAN,
            }

        except Exception as e:
            logger.error(f"Error predicting quality for {image_path}: {e}")
            return {
                "predicted_class": 0,
                "predicted_label": "contaminated",
                "confidence": 0.0,
                "contaminated_probability": 1.0,
                "clean_probability": 0.0,
                "review_probability": 0.0,
                "class_probabilities": {
                    "contaminated": 1.0,
                    "clean": 0.0,
                    "review": 0.0,
                },
                "is_clean": False,
                "error": str(e),
            }  # type: ignore[dict-item]

    def predict_batch(
        self,
        image_paths: list[str],
        fits_processor: Optional["FITSProcessor"] = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Predict quality for a batch of images.

        Args:
            image_paths: List of image paths
            fits_processor: Optional FITSProcessor for FITS files

        Returns:
            Dictionary mapping image paths to prediction results
        """
        results = {}

        for image_path in image_paths:
            results[image_path] = self.predict_single(image_path, fits_processor)

        return results


def create_data_transforms(train: bool = True) -> transforms.Compose:
    """
    Create data transforms for training or validation.

    Args:
        train: Whether this is for training (includes augmentation)

    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )


def _validate_label_thresholds(
    clean_threshold: float,
    contaminated_threshold: float,
) -> None:
    if not 0.0 <= contaminated_threshold <= 1.0:
        raise ValueError("contaminated_threshold must be between 0.0 and 1.0")
    if not 0.0 <= clean_threshold <= 1.0:
        raise ValueError("clean_threshold must be between 0.0 and 1.0")
    if contaminated_threshold >= clean_threshold:
        raise ValueError("contaminated_threshold must be lower than clean_threshold")


def _label_from_quality_score(
    quality_score: float,
    clean_threshold: float,
    contaminated_threshold: float,
) -> int:
    if quality_score >= clean_threshold:
        return LABEL_CLEAN
    if quality_score <= contaminated_threshold:
        return LABEL_CONTAMINATED
    return LABEL_REVIEW


def _label_id_from_name(label_name: str) -> int:
    try:
        return LABEL_IDS[label_name]
    except KeyError as exc:
        raise ValueError(f"Unknown label: {label_name}") from exc


def _generate_cv_label_records(
    fits_files: Sequence[Union[str, Path]],
    artifact_detector: Optional[Any] = None,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
    fits_processor: Optional["FITSProcessor"] = None,
) -> list[dict[str, Any]]:
    _validate_label_thresholds(clean_threshold, contaminated_threshold)

    if artifact_detector is None:
        from .cv_prefilter import ArtifactDetector

        artifact_detector = ArtifactDetector()

    if fits_processor is None:
        from .fits_processor import FITSProcessor

        fits_processor = FITSProcessor()

    records = []
    for fits_file in fits_files:
        path = Path(fits_file)
        processed = fits_processor.process_fits_file(path)
        if processed is None:
            logger.warning(f"Skipping unreadable FITS file: {path}")
            continue

        analysis = artifact_detector.comprehensive_analysis(
            processed["normalized_image"]
        )
        quality_score = float(analysis["overall_quality_score"])
        label = _label_from_quality_score(
            quality_score,
            clean_threshold,
            contaminated_threshold,
        )
        records.append(
            {
                "path": path,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "quality_score": quality_score,
                "needs_manual_review": bool(analysis["needs_manual_review"]),
            },
        )

    return records


def generate_training_labels_from_cv(
    fits_files: Sequence[Union[str, Path]],
    artifact_detector: Optional[Any] = None,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
    fits_processor: Optional["FITSProcessor"] = None,
) -> tuple[list[Path], list[int], list[Path]]:
    """
    Generate three-class training labels from CV quality scores.

    Labels are 0=contaminated, 1=clean, and 2=review. Review files are also
    returned separately so callers can inspect borderline cases.
    """
    records = _generate_cv_label_records(
        fits_files,
        artifact_detector,
        clean_threshold,
        contaminated_threshold,
        fits_processor,
    )
    labeled_files = [record["path"] for record in records]
    labels = [record["label"] for record in records]
    review_files = [
        record["path"] for record in records if record["label"] == LABEL_REVIEW
    ]
    return labeled_files, labels, review_files


def _write_manifest(
    manifest_path: Path, records: list[dict[str, Any]], split: str
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "split",
                "path",
                "label",
                "label_name",
                "quality_score",
                "needs_manual_review",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": split,
                    "path": str(record["path"]),
                    "label": record["label"],
                    "label_name": record["label_name"],
                    "quality_score": f"{record['quality_score']:.6f}",
                    "needs_manual_review": record["needs_manual_review"],
                },
            )


def _write_json_manifest(
    manifest_path: Path,
    records: list[dict[str, Any]],
    split: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_records = []
    for record in records:
        serializable_records.append(
            {
                "split": split,
                "path": str(record["path"]),
                "label": record["label"],
                "label_name": record["label_name"],
                "quality_score": record.get("quality_score"),
                "reviewed": record.get("reviewed", False),
            },
        )
    manifest_path.write_text(json.dumps({"files": serializable_records}, indent=2))


def create_dataset_from_cv_labels(
    fits_files: Sequence[Union[str, Path]],
    artifact_detector: Optional[Any] = None,
    output_dir: Union[str, Path] = "dataset",
    train_split: float = 0.8,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
    random_seed: int = 42,
) -> tuple[AstroImageDataset, AstroImageDataset, list[Path]]:
    """Create train/validation datasets from CV-generated FITS labels."""
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0.0 and 1.0")

    from .fits_processor import FITSProcessor

    output_path = Path(output_dir)
    fits_processor = FITSProcessor()
    records = _generate_cv_label_records(
        fits_files,
        artifact_detector,
        clean_threshold,
        contaminated_threshold,
        fits_processor,
    )
    if len(records) < 2:
        raise ValueError("At least two readable FITS files are required for training")

    random.Random(random_seed).shuffle(records)  # nosec B311 - deterministic split only
    split_index = int(len(records) * train_split)
    split_index = min(max(split_index, 1), len(records) - 1)

    train_records = records[:split_index]
    val_records = records[split_index:]
    review_files = [
        record["path"] for record in records if record["label"] == LABEL_REVIEW
    ]

    _write_manifest(output_path / "train_manifest.csv", train_records, "train")
    _write_manifest(output_path / "val_manifest.csv", val_records, "val")

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

    return train_dataset, val_dataset, review_files


def load_labeled_records_from_manifest(
    manifest_path: Union[str, Path],
    reviewed_only: bool = False,
) -> list[dict[str, Any]]:
    """Load labeled FITS records from a JSON batch manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    records = []

    for entry in manifest.get("files", []):
        if entry.get("error"):
            continue
        if reviewed_only and not entry.get("reviewed"):
            continue

        label_name = entry.get("corrected_label") or entry.get("decision_label")
        if label_name not in LABEL_IDS:
            logger.warning(f"Skipping manifest entry with unknown label: {label_name}")
            continue

        path_value = entry.get("destination_path") or entry.get("source_path")
        if not path_value:
            logger.warning("Skipping manifest entry without a source path")
            continue

        path = Path(path_value)
        if not path.exists():
            logger.warning(f"Skipping missing manifest file: {path}")
            continue

        records.append(
            {
                "path": path,
                "label": _label_id_from_name(str(label_name)),
                "label_name": str(label_name),
                "quality_score": entry.get("quality_score"),
                "reviewed": bool(entry.get("reviewed")),
            },
        )

    return records


def create_dataset_from_manifest(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path] = "dataset",
    train_split: float = 0.8,
    reviewed_only: bool = False,
    random_seed: int = 42,
) -> tuple[AstroImageDataset, AstroImageDataset, list[dict[str, Any]]]:
    """Create train/validation datasets from a reviewed JSON batch manifest."""
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0.0 and 1.0")

    from .fits_processor import FITSProcessor

    records = load_labeled_records_from_manifest(manifest_path, reviewed_only)
    if len(records) < 2:
        raise ValueError(
            "At least two labeled manifest files are required for training"
        )

    random.Random(random_seed).shuffle(records)  # nosec B311 - deterministic split only
    split_index = int(len(records) * train_split)
    split_index = min(max(split_index, 1), len(records) - 1)

    train_records = records[:split_index]
    val_records = records[split_index:]
    output_path = Path(output_dir)
    _write_json_manifest(output_path / "train_manifest.json", train_records, "train")
    _write_json_manifest(output_path / "val_manifest.json", val_records, "val")

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

    return train_dataset, val_dataset, records


def _find_fits_files(fits_directory: Union[str, Path]) -> list[Path]:
    fits_path = Path(fits_directory)
    patterns = ["*.fits", "*.fit", "*.fts"]
    fits_files: list[Path] = []
    for pattern in patterns:
        fits_files.extend(fits_path.rglob(pattern))
    return sorted(set(fits_files))


def complete_training_pipeline(
    fits_directory: Union[str, Path],
    model_output_path: Union[str, Path],
    dataset_output_dir: Union[str, Path],
    epochs: int = 20,
    batch_size: int = 32,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
    train_split: float = 0.8,
    device: Optional[str] = None,
    pretrained: bool = False,
) -> dict[str, Any]:
    """Run FITS discovery, CV labeling, dataset creation, and model training."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    fits_files = _find_fits_files(fits_directory)
    if len(fits_files) < 2:
        raise ValueError(f"At least two FITS files are required in {fits_directory}")

    train_dataset, val_dataset, review_files = create_dataset_from_cv_labels(
        fits_files=fits_files,
        output_dir=dataset_output_dir,
        train_split=train_split,
        clean_threshold=clean_threshold,
        contaminated_threshold=contaminated_threshold,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AstroQualityClassifier(num_classes=3, pretrained=pretrained)
    trainer = ModelTrainer(model, device=device)
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    trainer.save_model(model_output_path)

    label_counts = {
        LABEL_NAMES[label]: train_dataset.labels.count(label)
        + val_dataset.labels.count(label)
        for label in LABEL_NAMES
    }

    return {
        "model_path": str(model_output_path),
        "dataset_dir": str(dataset_output_dir),
        "history": history,
        "final_metrics": {
            "best_val_accuracy": (
                max(trainer.val_accuracies) if trainer.val_accuracies else 0.0
            ),
            "final_val_accuracy": (
                trainer.val_accuracies[-1] if trainer.val_accuracies else 0.0
            ),
            "final_val_loss": trainer.val_losses[-1] if trainer.val_losses else 0.0,
        },
        "dataset_stats": {
            "total_files": len(fits_files),
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "review_samples": len(review_files),
            "label_counts": label_counts,
        },
    }


def complete_training_pipeline_from_manifest(
    manifest_path: Union[str, Path],
    model_output_path: Union[str, Path],
    dataset_output_dir: Union[str, Path],
    epochs: int = 20,
    batch_size: int = 32,
    train_split: float = 0.8,
    reviewed_only: bool = False,
    device: Optional[str] = None,
    pretrained: bool = False,
) -> dict[str, Any]:
    """Train a three-class model from labels in a JSON batch manifest."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    train_dataset, val_dataset, records = create_dataset_from_manifest(
        manifest_path=manifest_path,
        output_dir=dataset_output_dir,
        train_split=train_split,
        reviewed_only=reviewed_only,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AstroQualityClassifier(num_classes=3, pretrained=pretrained)
    trainer = ModelTrainer(model, device=device)
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    trainer.save_model(model_output_path)

    all_labels = train_dataset.labels + val_dataset.labels
    label_counts = {
        LABEL_NAMES[label]: all_labels.count(label) for label in LABEL_NAMES
    }

    return {
        "model_path": str(model_output_path),
        "dataset_dir": str(dataset_output_dir),
        "manifest_path": str(manifest_path),
        "history": history,
        "final_metrics": {
            "best_val_accuracy": (
                max(trainer.val_accuracies) if trainer.val_accuracies else 0.0
            ),
            "final_val_accuracy": (
                trainer.val_accuracies[-1] if trainer.val_accuracies else 0.0
            ),
            "final_val_loss": trainer.val_losses[-1] if trainer.val_losses else 0.0,
        },
        "dataset_stats": {
            "total_manifest_records": len(records),
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "reviewed_samples": sum(1 for record in records if record.get("reviewed")),
            "label_counts": label_counts,
        },
    }


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimize a PyTorch model for inference.

    Args:
        model: PyTorch model to optimize

    Returns:
        Optimized model for faster inference
    """
    # Set to evaluation mode
    model.eval()

    # Apply quantization for CPU inference (with error handling for compatibility)
    try:
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )
        return model_quantized  # type: ignore[return-value,no-any-return]
    except (RuntimeError, AttributeError) as e:
        # Fallback for environments where quantization is not supported
        # (e.g., some PyTorch builds, certain platforms)
        print(f"Warning: Quantization not available ({e}), returning original model")
        return model
