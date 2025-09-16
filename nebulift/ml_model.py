"""
ResNet18 Model for Astrophotography Quality Assessment

Implements a ResNet18-based neural network for binary classification of
astronomical images as "clean" or "contaminated" with artifacts.
Optimized for CPU-only inference on resource-constrained devices like Raspberry Pi 5.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

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


class AstroImageDataset(Dataset):
    """Dataset class for astronomical images with quality labels."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
        fits_processor: Optional["FITSProcessor"] = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            image_paths: List of paths to image files
            labels: List of labels (0=contaminated, 1=clean)
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
        label = self.labels[idx]

        try:
            # Load image
            if self.fits_processor and image_path.endswith((".fits", ".fit", ".fts")):
                # Load FITS file
                processed_data = self.fits_processor.process_fits_file(Path(image_path))
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
    ) -> dict[str, Union[float, bool, str]]:
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

                contaminated_prob = probabilities[0][0].item()
                clean_prob = probabilities[0][1].item()

                predicted_class = torch.argmax(output, dim=1).item()
                confidence = max(contaminated_prob, clean_prob)

            return {
                "predicted_class": predicted_class,  # 0=contaminated, 1=clean
                "confidence": confidence,
                "contaminated_probability": contaminated_prob,
                "clean_probability": clean_prob,
                "is_clean": predicted_class == 1,
            }

        except Exception as e:
            logger.error(f"Error predicting quality for {image_path}: {e}")
            return {
                "predicted_class": 0,
                "confidence": 0.0,
                "contaminated_probability": 1.0,
                "clean_probability": 0.0,
                "is_clean": False,
                "error": str(e),
            }  # type: ignore[dict-item]

    def predict_batch(
        self,
        image_paths: list[str],
        fits_processor: Optional["FITSProcessor"] = None,
    ) -> dict[str, dict[str, Union[float, bool, str]]]:
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
