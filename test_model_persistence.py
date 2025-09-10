"""
Test Model Persistence and Checkpointing

This module tests the model persistence functionality including saving,
loading, and versioning of trained models.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from nebulift.ml_model import AstroQualityClassifier, ModelTrainer
from nebulift.model_persistence import ModelCheckpoint, ModelVersioning


class TestModelPersistence(unittest.TestCase):
    """Test model persistence functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = AstroQualityClassifier(num_classes=2, pretrained=False)
        self.trainer = ModelTrainer(self.model, device="cpu")

        # Add some mock training history
        self.trainer.train_losses = [0.8, 0.6, 0.4]
        self.trainer.val_losses = [0.7, 0.5, 0.3]
        self.trainer.train_accuracies = [60.0, 70.0, 80.0]
        self.trainer.val_accuracies = [65.0, 75.0, 85.0]

    def test_save_and_load_model(self):
        """Test basic save and load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"

            # Save model
            self.trainer.save_model(model_path)
            self.assertTrue(model_path.exists())

            # Check metadata file was created
            metadata_path = model_path.with_suffix(".json")
            self.assertTrue(metadata_path.exists())

            # Load model
            loaded_trainer = ModelTrainer.load_model(model_path, device="cpu")

            # Verify model architecture
            self.assertEqual(type(loaded_trainer.model), type(self.trainer.model))

            # Verify training history was restored
            self.assertEqual(loaded_trainer.train_losses, self.trainer.train_losses)
            self.assertEqual(loaded_trainer.val_losses, self.trainer.val_losses)
            self.assertEqual(
                loaded_trainer.train_accuracies,
                self.trainer.train_accuracies,
            )
            self.assertEqual(loaded_trainer.val_accuracies, self.trainer.val_accuracies)

    def test_load_model_for_inference(self):
        """Test loading model for inference only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "inference_model.pth"

            # Save model
            self.trainer.save_model(model_path)

            # Load for inference
            inference_model = ModelCheckpoint.load_model_for_inference(
                model_path,
                device="cpu",
            )

            # Verify model is in eval mode
            self.assertFalse(inference_model.training)

            # Verify model can perform inference
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = inference_model(test_input)
                self.assertEqual(output.shape, (1, 2))

    def test_get_model_info(self):
        """Test retrieving model information without loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "info_model.pth"

            # Save model
            self.trainer.save_model(model_path)

            # Get model info
            info = ModelCheckpoint.get_model_info(model_path)

            # Verify information is present
            self.assertIn("model_class", info)
            self.assertIn("save_date", info)
            self.assertIn("total_epochs", info)
            self.assertIn("best_val_accuracy", info)
            self.assertEqual(info["model_class"], "AstroQualityClassifier")
            self.assertEqual(info["total_epochs"], 3)
            self.assertEqual(info["best_val_accuracy"], 85.0)

    def test_save_without_metadata(self):
        """Test saving model without metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "no_metadata_model.pth"

            # Save model without metadata
            self.trainer.save_model(model_path, include_metadata=False)
            self.assertTrue(model_path.exists())

            # Metadata file should not exist
            metadata_path = model_path.with_suffix(".json")
            self.assertFalse(metadata_path.exists())

            # Load model
            loaded_trainer = ModelTrainer.load_model(model_path, device="cpu")

            # Training history should be empty since no metadata was saved
            self.assertEqual(loaded_trainer.train_losses, [])
            self.assertEqual(loaded_trainer.val_losses, [])

    def test_save_without_training_state(self):
        """Test saving model without optimizer/scheduler state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "no_training_state.pth"

            # Save model without training state
            self.trainer.save_model(model_path, include_training_state=False)

            # Load and verify
            checkpoint = torch.load(model_path, map_location="cpu")
            self.assertNotIn("optimizer_state_dict", checkpoint)
            self.assertNotIn("scheduler_state_dict", checkpoint)

    def test_file_not_found_error(self):
        """Test error handling for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            ModelTrainer.load_model("non_existent_file.pth")

        with self.assertRaises(FileNotFoundError):
            ModelCheckpoint.get_model_info("non_existent_file.pth")


class TestModelVersioning(unittest.TestCase):
    """Test model versioning functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = AstroQualityClassifier(num_classes=2, pretrained=False)
        self.trainer = ModelTrainer(self.model, device="cpu")

        # Add some mock training history
        self.trainer.train_losses = [0.5, 0.3]
        self.trainer.val_losses = [0.4, 0.2]
        self.trainer.train_accuracies = [70.0, 90.0]
        self.trainer.val_accuracies = [75.0, 95.0]

    def test_save_versioned_model(self):
        """Test saving model with automatic versioning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            versioning = ModelVersioning(temp_dir)

            # Save versioned model
            model_path = versioning.save_versioned_model(
                self.trainer,
                "test_model",
                version="v1.0",
                tags={"experiment": "test", "dataset": "mock"},
            )

            # Verify model was saved
            self.assertTrue(Path(model_path).exists())

            # Verify registry was updated
            registry = versioning.list_models()
            self.assertIn("test_model", registry)
            self.assertIn("v1.0", registry["test_model"]["versions"])

    def test_load_latest_model(self):
        """Test loading latest version of a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            versioning = ModelVersioning(temp_dir)

            # Save multiple versions
            versioning.save_versioned_model(self.trainer, "test_model", version="v1.0")
            versioning.save_versioned_model(self.trainer, "test_model", version="v2.0")

            # Load latest version
            loaded_trainer = versioning.load_latest_model("test_model", device="cpu")

            # Verify model was loaded correctly
            self.assertEqual(type(loaded_trainer.model), type(self.trainer.model))
            self.assertEqual(loaded_trainer.train_losses, self.trainer.train_losses)

    def test_list_models(self):
        """Test listing all registered models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            versioning = ModelVersioning(temp_dir)

            # Save multiple models
            versioning.save_versioned_model(self.trainer, "model_a", version="v1.0")
            versioning.save_versioned_model(self.trainer, "model_b", version="v1.0")

            # List models
            registry = versioning.list_models()

            # Verify both models are present
            self.assertIn("model_a", registry)
            self.assertIn("model_b", registry)
            self.assertEqual(len(registry), 2)

    def test_model_not_found_error(self):
        """Test error handling for non-existent models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            versioning = ModelVersioning(temp_dir)

            with self.assertRaises(ValueError):
                versioning.load_latest_model("non_existent_model")

    def test_automatic_version_generation(self):
        """Test automatic version generation when not specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            versioning = ModelVersioning(temp_dir)

            # Save model without specifying version
            model_path = versioning.save_versioned_model(
                self.trainer,
                "auto_version_model",
            )

            # Verify version was auto-generated
            registry = versioning.list_models()
            versions = list(registry["auto_version_model"]["versions"].keys())
            self.assertEqual(len(versions), 1)

            # Version should be in YYYYMMDD_HHMMSS format
            version = versions[0]
            self.assertEqual(len(version), 15)  # YYYYMMDD_HHMMSS
            self.assertIn("_", version)


class TestModelPersistenceIntegration(unittest.TestCase):
    """Integration tests for model persistence with training pipeline."""

    def test_end_to_end_persistence_workflow(self):
        """Test complete save/load workflow with training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and train a small model
            model = AstroQualityClassifier(num_classes=2, pretrained=False)
            trainer = ModelTrainer(model, device="cpu")

            # Simulate training history
            trainer.train_losses = [0.8, 0.5, 0.3]
            trainer.val_losses = [0.7, 0.4, 0.2]
            trainer.train_accuracies = [60.0, 75.0, 90.0]
            trainer.val_accuracies = [65.0, 80.0, 95.0]

            # Save model
            model_path = Path(temp_dir) / "trained_model.pth"
            trainer.save_model(model_path)

            # Load model in new trainer instance
            loaded_trainer = ModelTrainer.load_model(model_path, device="cpu")

            # Verify complete state restoration
            self.assertEqual(loaded_trainer.train_losses, trainer.train_losses)
            self.assertEqual(loaded_trainer.val_losses, trainer.val_losses)
            self.assertEqual(loaded_trainer.train_accuracies, trainer.train_accuracies)
            self.assertEqual(loaded_trainer.val_accuracies, trainer.val_accuracies)

            # Verify model can continue training (optimizer state restored)
            original_lr = trainer.optimizer.param_groups[0]["lr"]
            loaded_lr = loaded_trainer.optimizer.param_groups[0]["lr"]
            self.assertEqual(original_lr, loaded_lr)

    def test_model_compatibility_across_versions(self):
        """Test that models saved with different configurations can be loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different model configurations
            configurations = [
                {"num_classes": 2, "dropout_rate": 0.2},
                {"num_classes": 3, "dropout_rate": 0.5},
            ]

            for i, config in enumerate(configurations):
                model = AstroQualityClassifier(
                    num_classes=config["num_classes"],
                    dropout_rate=config["dropout_rate"],
                    pretrained=False,
                )
                trainer = ModelTrainer(model, device="cpu")
                trainer.train_losses = [0.5]
                trainer.val_losses = [0.4]
                trainer.train_accuracies = [80.0]
                trainer.val_accuracies = [85.0]

                # Save model
                model_path = Path(temp_dir) / f"model_config_{i}.pth"
                trainer.save_model(model_path)

                # Load and verify
                loaded_trainer = ModelTrainer.load_model(model_path, device="cpu")
                self.assertEqual(
                    loaded_trainer.model.backbone.fc.out_features,
                    config["num_classes"],
                )
                self.assertAlmostEqual(
                    loaded_trainer.model.dropout.p,
                    config["dropout_rate"],
                    places=3,
                )


if __name__ == "__main__":
    unittest.main()
