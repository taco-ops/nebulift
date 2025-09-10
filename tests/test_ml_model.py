"""
Test suite for ML Model

Tests the ResNet18-based quality classifier and related components.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn

from nebulift.ml_model import (
    AstroImageDataset,
    AstroQualityClassifier,
    ModelTrainer,
    QualityPredictor,
    create_data_transforms,
    optimize_model_for_inference,
)


class TestAstroImageDataset:
    """Test cases for AstroImageDataset class."""

    def test_init(self):
        """Test dataset initialization."""
        image_paths = ["image1.jpg", "image2.jpg"]
        labels = [0, 1]

        dataset = AstroImageDataset(image_paths, labels)

        assert len(dataset) == 2
        assert dataset.image_paths == image_paths
        assert dataset.labels == labels

    def test_init_mismatched_lengths(self):
        """Test dataset initialization with mismatched lengths."""
        image_paths = ["image1.jpg", "image2.jpg"]
        labels = [0]  # Fewer labels than images

        with pytest.raises(AssertionError):
            AstroImageDataset(image_paths, labels)

    @patch("nebulift.ml_model.Image.open")
    def test_getitem_regular_image(self, mock_image_open):
        """Test getting item with regular image file."""
        # Mock PIL Image
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image

        # Mock transform
        mock_transform = Mock()
        mock_transform.return_value = torch.zeros((3, 224, 224))

        dataset = AstroImageDataset(["test.jpg"], [1], transform=mock_transform)

        tensor, label = dataset[0]

        assert isinstance(tensor, torch.Tensor)
        assert label == 1
        mock_image_open.assert_called_once_with("test.jpg")

    def test_getitem_fits_file(self):
        """Test getting item with FITS file."""
        # Mock FITS processor
        mock_processor = Mock()
        mock_processed_data = {
            "ml_image": np.random.random((224, 224)),
        }
        mock_processor.process_fits_file.return_value = mock_processed_data

        # Mock transform
        mock_transform = Mock()
        mock_transform.return_value = torch.zeros((3, 224, 224))

        dataset = AstroImageDataset(
            ["test.fits"], [0],
            transform=mock_transform,
            fits_processor=mock_processor,
        )

        tensor, label = dataset[0]

        assert isinstance(tensor, torch.Tensor)
        assert label == 0

    def test_getitem_error_handling(self):
        """Test error handling when image loading fails."""
        dataset = AstroImageDataset(["nonexistent.jpg"], [1])

        # Should return dummy tensor and label 0 on error
        tensor, label = dataset[0]

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert label == 0


class TestAstroQualityClassifier:
    """Test cases for AstroQualityClassifier model."""

    def test_init_pretrained(self):
        """Test model initialization with pretrained weights."""
        model = AstroQualityClassifier(num_classes=2, pretrained=True)

        assert isinstance(model, nn.Module)
        assert isinstance(model.backbone, torch.nn.modules.container.Sequential) or hasattr(model.backbone, "fc")
        assert hasattr(model, "dropout")

    def test_init_not_pretrained(self):
        """Test model initialization without pretrained weights."""
        model = AstroQualityClassifier(num_classes=2, pretrained=False)

        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = AstroQualityClassifier(num_classes=2, pretrained=False)

        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        output = model(input_tensor)

        assert output.shape == (batch_size, 2)
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_single_image(self):
        """Test forward pass with single image."""
        model = AstroQualityClassifier(num_classes=2, pretrained=False)

        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)

        assert output.shape == (1, 2)


class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = AstroQualityClassifier(num_classes=2, pretrained=False)
        self.trainer = ModelTrainer(self.model, device="cpu", learning_rate=1e-3)

    def test_init(self):
        """Test trainer initialization."""
        assert self.trainer.device == "cpu"
        assert isinstance(self.trainer.criterion, nn.CrossEntropyLoss)
        assert isinstance(self.trainer.optimizer, torch.optim.Adam)

    def test_train_epoch(self):
        """Test training for one epoch."""
        # Create dummy data loader
        dataset = [(torch.randn(3, 224, 224), torch.randint(0, 2, (1,)).item()) for _ in range(10)]
        data_loader = [(torch.stack([item[0] for item in dataset[:2]]),
                       torch.tensor([item[1] for item in dataset[:2]])) for _ in range(5)]

        loss, accuracy = self.trainer.train_epoch(data_loader)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100

    def test_evaluate(self):
        """Test model evaluation."""
        # Create dummy data loader
        dataset = [(torch.randn(3, 224, 224), torch.randint(0, 2, (1,)).item()) for _ in range(10)]
        data_loader = [(torch.stack([item[0] for item in dataset[:2]]),
                       torch.tensor([item[1] for item in dataset[:2]])) for _ in range(5)]

        loss, accuracy = self.trainer.evaluate(data_loader)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100


class TestQualityPredictor:
    """Test cases for QualityPredictor class."""

    @patch("nebulift.ml_model.torch.load")
    @patch.object(AstroQualityClassifier, "load_state_dict")
    def test_init(self, mock_load_state_dict, mock_torch_load):
        """Test predictor initialization."""
        # Mock loading state dict
        mock_state_dict = {}
        mock_torch_load.return_value = mock_state_dict

        predictor = QualityPredictor("dummy_model.pth", device="cpu")

        assert predictor.device == "cpu"
        assert isinstance(predictor.model, AstroQualityClassifier)
        mock_load_state_dict.assert_called_once()

    @patch("nebulift.ml_model.torch.load")
    @patch.object(AstroQualityClassifier, "load_state_dict")
    @patch("nebulift.ml_model.Image.open")
    def test_predict_single_regular_image(self, mock_image_open, mock_load_state_dict, mock_torch_load):
        """Test single image prediction with regular image."""
        # Mock model loading
        mock_torch_load.return_value = {}

        # Mock PIL Image
        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image

        # Create predictor
        predictor = QualityPredictor("dummy_model.pth")

        # Mock model output
        with patch.object(predictor.model, "forward") as mock_forward:
            mock_output = torch.tensor([[0.3, 0.7]])  # Clean prediction
            mock_forward.return_value = mock_output

            result = predictor.predict_single("test.jpg")

            assert isinstance(result, dict)
            assert "predicted_class" in result
            assert "confidence" in result
            assert "is_clean" in result
            assert result["predicted_class"] in [0, 1]

    @patch("nebulift.ml_model.torch.load")
    @patch.object(AstroQualityClassifier, "load_state_dict")
    def test_predict_single_error_handling(self, mock_load_state_dict, mock_torch_load):
        """Test error handling in single prediction."""
        mock_torch_load.return_value = {}

        predictor = QualityPredictor("dummy_model.pth")

        # Test with nonexistent file
        result = predictor.predict_single("nonexistent.jpg")

        assert isinstance(result, dict)
        assert "error" in result
        assert result["predicted_class"] == 0
        assert result["is_clean"] == False

    @patch("nebulift.ml_model.torch.load")
    @patch.object(AstroQualityClassifier, "load_state_dict")
    def test_predict_batch(self, mock_load_state_dict, mock_torch_load):
        """Test batch prediction."""
        mock_torch_load.return_value = {}

        predictor = QualityPredictor("dummy_model.pth")

        # Mock predict_single method
        with patch.object(predictor, "predict_single") as mock_predict:
            mock_predict.return_value = {
                "predicted_class": 1,
                "confidence": 0.8,
                "is_clean": True,
            }

            image_paths = ["image1.jpg", "image2.jpg"]
            results = predictor.predict_batch(image_paths)

            assert len(results) == 2
            assert all(path in results for path in image_paths)
            assert mock_predict.call_count == 2


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_create_data_transforms_train(self):
        """Test creation of training transforms."""
        transforms = create_data_transforms(train=True)

        assert transforms is not None
        # Should have more transforms for training (augmentation)
        assert len(transforms.transforms) > 5

    def test_create_data_transforms_val(self):
        """Test creation of validation transforms."""
        transforms = create_data_transforms(train=False)

        assert transforms is not None
        # Should have fewer transforms for validation
        assert len(transforms.transforms) <= 5

    def test_optimize_model_for_inference(self):
        """Test model optimization for inference."""
        model = AstroQualityClassifier(num_classes=2, pretrained=False)

        optimized_model = optimize_model_for_inference(model)

        assert optimized_model is not None
        # Note: The actual quantization might not work in test environment
        # but the function should not crash


class TestIntegration:
    """Integration tests for ML model components."""

    def test_end_to_end_training_pipeline(self):
        """Test basic end-to-end training pipeline."""
        # Create dummy dataset
        image_paths = ["dummy1.jpg", "dummy2.jpg", "dummy3.jpg", "dummy4.jpg"]
        labels = [0, 1, 0, 1]

        # Mock image loading for dataset
        with patch("nebulift.ml_model.Image.open") as mock_open:
            mock_image = Mock()
            mock_image.mode = "RGB"
            mock_open.return_value = mock_image

            # Create transforms
            transform = create_data_transforms(train=True)

            # Mock transform to return dummy tensor
            with patch.object(transform, "__call__") as mock_transform:
                mock_transform.return_value = torch.randn(3, 224, 224)

                # Create dataset
                dataset = AstroImageDataset(image_paths, labels, transform=transform)

                # Test dataset loading
                assert len(dataset) == 4
                tensor, label = dataset[0]
                assert isinstance(tensor, torch.Tensor)
                assert label in [0, 1]

    def test_model_compatibility(self):
        """Test model compatibility with different input sizes."""
        model = AstroQualityClassifier(num_classes=2, pretrained=False)

        # Test with standard input
        input_224 = torch.randn(1, 3, 224, 224)
        output = model(input_224)
        assert output.shape == (1, 2)

        # Model should handle batch sizes
        input_batch = torch.randn(8, 3, 224, 224)
        output_batch = model(input_batch)
        assert output_batch.shape == (8, 2)
